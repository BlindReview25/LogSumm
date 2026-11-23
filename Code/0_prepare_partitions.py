from decimal import Decimal, InvalidOperation
import argparse
import csv
import hashlib
import json
import math
import os
import random
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

# Defaults can be edited here; command-line arguments override them.
DEFAULT_WINDOW_SIZE = 10 # 5, 10, 15
DEFAULT_RATIOS = [0.75, 0.10, 0.15]  # train, validation, test
DEFAULT_SIMILARITY_THRESHOLD = 0.70
DEFAULT_ANOMALY_TO_NORMAL_RATIO = 1.0
DEFAULT_SEED = 42
SPLITS = ['train', 'validation', 'test']
CATEGORIES = ['seen', 'unseen']

def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))

def setup(db):
    db.executescript('''
      PRAGMA journal_mode=DELETE;
      PRAGMA temp_store=FILE;
      PRAGMA cache_size=-32768;
      CREATE TABLE segments (
        id INTEGER PRIMARY KEY, label TEXT NOT NULL, n INTEGER NOT NULL,
        anomalies INTEGER NOT NULL, first_row INTEGER NOT NULL, last_row INTEGER NOT NULL,
        sequence TEXT NOT NULL, payload TEXT NOT NULL, rank INTEGER NOT NULL,
        split TEXT, retained INTEGER DEFAULT 1, reason TEXT DEFAULT '', representative INTEGER
      );
      CREATE TABLE train_sequences (sequence TEXT PRIMARY KEY) WITHOUT ROWID;
      CREATE TABLE train_templates (template TEXT PRIMARY KEY) WITHOUT ROWID;
    ''')
    db.execute("ALTER TABLE segments ADD COLUMN anomaly_signature TEXT NOT NULL DEFAULT '[]'")


def allocate(n, ratios):
    exact = [n * r for r in ratios]
    sizes = [math.floor(x) for x in exact]
    for i in sorted(range(3), key=lambda j: (-(exact[j] - sizes[j]), j))[:n-sum(sizes)]:
        sizes[i] += 1
    if min(sizes) == 0:
        raise ValueError(f'A class with {n} segments cannot populate all partitions under these ratios. Increase input/window coverage or change ratios; no silent fallback is used.')
    return sizes

def split_segments(db, a):
    counts = dict(db.execute('SELECT label,count(*) FROM segments GROUP BY label'))
    if set(counts) != ({'unknown'} if a.annotation_mode=='unknown' else {'normal', 'anomalous'}):
        raise ValueError(f'Both classes are required for this experiment; found {counts}.')
    db.execute('CREATE INDEX label_rank ON segments(label,rank,id)')
    for label, n in sorted(counts.items()):
        sizes = allocate(n, a.ratios)
        cursor = db.execute('SELECT id FROM segments WHERE label=? ORDER BY rank,id', (label,))
        limits = [sizes[0], sizes[0] + sizes[1]]
        position = 0
        while True:
            batch = cursor.fetchmany(10000)
            if not batch:
                break
            updates = []
            for (i,) in batch:
                part = 0 if position < limits[0] else 1 if position < limits[1] else 2
                updates.append((SPLITS[part], i))
                position += 1
            db.executemany('UPDATE segments SET split=? WHERE id=?', updates)
    db.commit()
    db.execute('CREATE INDEX split_label_rank ON segments(split,label,rank,id)')
    print(('Random' if a.annotation_mode=='unknown' else 'Stratified') + ' split complete (pattern overlap allowed).', flush=True)

def build_reference(db):
    db.execute("INSERT OR IGNORE INTO train_sequences SELECT sequence FROM segments WHERE split='train' AND retained=1")
    for (seq,) in db.execute('SELECT sequence FROM train_sequences'):
        db.executemany('INSERT OR IGNORE INTO train_templates VALUES(?)', ((x,) for x in set(json.loads(seq))))
    db.commit()

def segment_id(i):
    return f'chunk_{i:09d}' if i is not None else ''

def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def ingest(db, a):
    rng = random.Random(a.seed)
    headers = []
    block = []
    event_count = 0
    previous_line_id = None
    previous_timestamp = None
    a.annotation_mode = None
    a.timestamp_order_checked = False
    a.timestamp_inversions = 0
    a.line_id_gaps = 0
    segment_count = 0
    def save(rows, last_row):
        nonlocal segment_count
        segment_count += 1
        first = last_row - len(rows) + 1
        labels = [r[a.label_column].strip() for r in rows]
        templates = [r[a.template_column] for r in rows]
        count = sum(x != a.normal_label.strip() for x in labels) if a.annotation_mode=='annotated' else 0
        anomaly_signature=dumps(sorted({t for t,l in zip(templates,labels) if a.annotation_mode=='annotated' and l != a.normal_label.strip()}))
        content, labeled = [], []
        for number, r in enumerate(rows, first):
            identifier = r.get(a.id_column, '').strip() or str(number)
            body = ' '.join(r[c] for c in headers if r[c] != '')
            body = (body + ' ' if body else '') + r[a.content_column]
            content.append(f'[id={identifier}] {body}')
            labeled.append(f'[id={identifier}] [label={r[a.label_column]}] {body}')
        payload = {'Content': '\n'.join(content), 'Content_with_labels': '\n'.join(labeled),
                   'messages_only': '\n'.join(r[a.content_column] for r in rows),
                   'event_labels_json': dumps(labels), 'events_json': dumps(rows)}
        db.execute('INSERT INTO segments(id,label,n,anomalies,first_row,last_row,sequence,payload,rank,anomaly_signature) VALUES(?,?,?,?,?,?,?,?,?,?)',
                   (segment_count, 'unknown' if a.annotation_mode=='unknown' else ('anomalous' if count else 'normal'), len(rows), count, first, last_row,
                    dumps(templates), dumps(payload), rng.getrandbits(63), anomaly_signature))
        if segment_count % 10000 == 0:
            db.commit()
            print(f'Created {segment_count:,} segments from {last_row:,} events', flush=True)
    with a.input.open(encoding=a.encoding, newline='') as f:
        reader = csv.DictReader(f, delimiter=a.delimiter)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise ValueError('Duplicate CSV column names.')
        required = {a.content_column, a.template_column}
        if required - set(fields):
            raise ValueError(f'Missing required columns: {sorted(required - set(fields))}. Provide an event-level parsed CSV containing Content and EventTemplate.')
        a.input_has_label = a.label_column in fields
        if not a.input_has_label:
            print(f'No {a.label_column!r} column found: treating all source events as unknown (unannotated mode).', flush=True)
        headers = [c.strip() for c in a.header_columns.split(',') if c.strip() in fields]
        if required & set(headers):
            raise ValueError('Header columns must not include label, message, or template columns.')
        for event_count, row in enumerate(reader, 1):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed CSV record {event_count}: inconsistent column count.')
            if any(not row[c].strip() for c in required):
                raise ValueError(f'Blank required value in CSV record {event_count}.')
            if not a.input_has_label:
                row[a.label_column] = 'unknown'
            elif not row[a.label_column].strip():
                raise ValueError(f'Blank {a.label_column} in CSV record {event_count}. For an unannotated dataset, omit the entire label column or fill every row with unknown.')
            mode = 'unknown' if row[a.label_column].strip().lower()=='unknown' else 'annotated'
            if a.annotation_mode is None:
                a.annotation_mode = mode
                if mode=='unknown' and a.auto_header_columns:
                    headers = [c for c in ['Timestamp','Date','Month','Day','Node','Host','Time','Type','Location','Thread','Component','PID','Level','RequestContext'] if c in fields]
                if not a.skip_time_order_check and a.timestamp_column not in fields and mode=='annotated':
                    raise ValueError(f'Missing timestamp column {a.timestamp_column}. Set --timestamp-column or --skip-time-order-check.')
                a.timestamp_order_checked = not a.skip_time_order_check and a.timestamp_column in fields
            elif a.annotation_mode != mode:
                raise ValueError(f'Mixed unknown and annotated labels at record {event_count}; supply one annotation mode per dataset.')
            if mode=='unknown':
                row[a.label_column] = 'unknown'
            if a.id_column in fields:
                try:
                    current_id = int(row[a.id_column])
                except ValueError:
                    raise ValueError(f'LineId must be an integer at CSV record {event_count}.')
                if previous_line_id is not None and current_id != previous_line_id + 1:
                    a.line_id_gaps += 1
                    if not a.allow_line_id_gaps:
                        raise ValueError(f'Nonconsecutive LineId at CSV record {event_count}: {previous_line_id} -> {current_id}. Resolve rejected logs or explicitly use --allow-line-id-gaps.')
                previous_line_id = current_id
            if a.timestamp_order_checked:
                try:
                    timestamp=Decimal(row[a.timestamp_column])
                    if not timestamp.is_finite():raise InvalidOperation
                except (InvalidOperation,ValueError):
                    raise ValueError(f'Invalid numeric timestamp at record {event_count}.')
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    a.timestamp_inversions += 1
                    raise ValueError(f'Timestamp order decreases at CSV record {event_count}. Sort numerically by {a.timestamp_column} with stable tie ordering before segmentation. No automatic reordering was performed.')
                previous_timestamp=timestamp
            block.append(row)
            if len(block) == a.window_size:
                save(block, event_count)
                block = []
        dropped = len(block) if a.drop_incomplete else 0
        if block and not a.drop_incomplete:
            save(block, event_count)
    db.commit()
    if not segment_count:
        raise ValueError('No segments created.')
    return event_count, dropped, headers

DATASET_NAMES = {'bgl':'BGL', 'thunderbird':'Thunderbird', 'apache':'Apache',
                 'linux':'Linux', 'openstack':'OpenStack', 'zookeeper':'Zookeeper',
                 'proxifier':'Proxifier'}


def partition_path(a, role):
    return a.output / f'{a.run_name}_{role}.csv'


def existing_partitions(folder):
    """Resolve one consistent set, accepting both legacy and prefixed filenames."""
    candidates = []
    legacy = folder / 'train_unique.csv'
    if legacy.is_file():
        candidates.append('')
    candidates.extend(p.name[:-len('train_unique.csv')] for p in folder.glob('*_train_unique.csv'))
    complete = [prefix for prefix in candidates
                if all((folder / (prefix + role + '.csv')).is_file()
                       for role in ['train_unique', 'validation', 'test'])]
    if len(complete) != 1:
        raise ValueError('Expected exactly one complete train_unique/validation/test file set (legacy or matching prefixed names).')
    return [folder / (complete[0] + role + '.csv') for role in ['train_unique','validation','test']]


def arguments():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--analyze-existing',type=Path,help='Analyze existing train_unique.csv, validation.csv and test.csv without rerunning sampling. Use --output NEW_FOLDER.')
    p.add_argument('--input',type=Path,help='Auto-detect one *structured*.csv in current folder if omitted.')
    p.add_argument('--dataset',choices=list(DATASET_NAMES),default=None,help='Inferred from input filename when omitted; specify for custom filenames.')
    p.add_argument('--output',type=Path,help='Optional exact output directory; default is the generated run name in the current directory.')
    p.add_argument('--include-threshold-in-name',action='store_true',help='Append similarity method and threshold to the run name.')
    p.add_argument('--work-dir',type=Path,help='Optional directory on a drive with ample free space.')
    p.add_argument('--window-size',type=int,default=DEFAULT_WINDOW_SIZE)
    p.add_argument('--ratios',nargs=3,type=float,default=DEFAULT_RATIOS,metavar=('TRAIN','VAL','TEST'))
    p.add_argument('--timestamp-column',default='Timestamp')
    p.add_argument('--skip-time-order-check',action='store_true',help='Explicitly accept source CSV order without numeric timestamp validation.')
    p.add_argument('--seed',type=int,default=DEFAULT_SEED)
    p.add_argument('--similarity-method',choices=['jaccard','multiset_jaccard'],default='jaccard')
    p.add_argument('--threshold',type=float,default=DEFAULT_SIMILARITY_THRESHOLD)
    p.add_argument('--anomaly-to-normal-ratio',type=float,default=DEFAULT_ANOMALY_TO_NORMAL_RATIO,
                   help='Minimum training anomaly:normal ratio, achieved by oversampling anomalies only.')
    p.add_argument('--no-oversampling',action='store_true')
    p.add_argument('--label-conflicts',choices=['error','exclude'],default='error')
    p.add_argument('--drop-incomplete',action='store_true')
    p.add_argument('--allow-line-id-gaps',action='store_true')
    p.add_argument('--keep-work-db',action='store_true')
    p.add_argument('--encoding',default='utf-8-sig')
    p.add_argument('--delimiter',default=',')
    p.add_argument('--label-column',default='Label')
    p.add_argument('--content-column',default='Content')
    p.add_argument('--template-column',default='EventTemplate')
    p.add_argument('--normal-label',default='-')
    p.add_argument('--id-column',default='LineId')
    p.add_argument('--header-columns',default=None)
    a=p.parse_args()
    if a.analyze_existing is not None:
        a.analyze_existing=a.analyze_existing.resolve()
        a.output=(a.output or a.analyze_existing.with_name(a.analyze_existing.name+'_novelty')).resolve()
        try:existing_partitions(a.analyze_existing)
        except ValueError as exc:p.error(str(exc))
        if a.output.exists() and (not a.output.is_dir() or any(a.output.iterdir())):
            p.error('Analysis output must be a new or empty directory.')
        return a
    if a.input is None:
        # Prefer a single raw parsed event dataset (e.g., Apache_full_parsed.csv).
        # Never auto-detect an already-generated train/validation/test partition.
        candidates=sorted(Path.cwd().glob('*_full_parsed.csv'))
        if not candidates:
            candidates=sorted(Path.cwd().glob('*structured*.csv'))
        if len(candidates)!=1:
            p.error('Specify --input: expected exactly one *_full_parsed.csv or *structured*.csv event-level file in the current folder.')
        a.input=candidates[0]
    if a.window_size<1 or any(r<=0 for r in a.ratios) or not math.isclose(sum(a.ratios),1):
        p.error('Positive window size and three positive split ratios summing to one are required.')
    if not 0<a.threshold<=1 or not math.isfinite(a.anomaly_to_normal_ratio) or a.anomaly_to_normal_ratio<=0:
        p.error('threshold must be in (0,1]; oversampling ratio must be finite and positive.')
    if len(a.delimiter)!=1:p.error('Delimiter must be one character.')
    a.input=a.input.resolve()
    if a.dataset is None:
        matches=[name for name in DATASET_NAMES if name in a.input.name.lower()]
        if len(matches)!=1:p.error('Cannot infer dataset name; specify --dataset.')
        a.dataset=matches[0]
    percentages='-'.join(format(Decimal(str(r))*100, 'f').rstrip('0').rstrip('.') if '.' in format(Decimal(str(r))*100,'f') else format(Decimal(str(r))*100,'f') for r in a.ratios)
    a.run_name=f'{DATASET_NAMES[a.dataset]}_split{percentages}_w{a.window_size}_seed{a.seed}'
    if a.include_threshold_in_name:
        a.run_name+=f'_{a.similarity_method}_sim{a.threshold:g}'
    a.output=(a.output or Path(a.run_name)).resolve()
    if not a.input.is_file():p.error(f'Input not found: {a.input}')
    if a.output.exists() and (not a.output.is_dir() or any(a.output.iterdir())):
        p.error('Output directory must be new or empty. Select another --output directory.')
    a.auto_header_columns = a.header_columns is None
    if a.header_columns is None:
        a.header_columns=('Timestamp,Date,Node,Time,Type,Component,Level' if a.dataset=='bgl'
                          else 'Date,Time,Process,Destination' if a.dataset=='proxifier'
                          else 'Timestamp,Date,Node,Location,Component,PID')
    return a

def detect_conflicts(db,a):
    # Global check catches conflicts within AND across partitions. It is only a
    # data-consistency check; partition assignment has already happened.
    db.execute('CREATE TABLE conflicts AS SELECT sequence,count(*) n,sum(label=\'normal\') normal_count,sum(label=\'anomalous\') anomalous_count FROM segments GROUP BY sequence HAVING count(DISTINCT label)>1')
    db.execute('CREATE UNIQUE INDEX conflict_sequence ON conflicts(sequence)')
    count=db.execute('SELECT count(*) FROM conflicts').fetchone()[0]
    fields=['templates_json','occurrences','normal_count','anomalous_count']
    write_csv(a.output/'label_conflicts.csv',fields,
              (dict(zip(fields,row)) for row in db.execute('SELECT * FROM conflicts ORDER BY sequence')))
    if count and a.label_conflicts=='error':
        raise ValueError(f'{count} exact template sequences have conflicting labels. See label_conflicts.csv; no partitions exported. Inspect conflicts, or rerun in a NEW output folder with --label-conflicts exclude to exclude ALL their occurrences.')
    if count:
        db.execute("UPDATE segments SET retained=0,reason='conflicting_labels_excluded' WHERE sequence IN (SELECT sequence FROM conflicts)")
    db.commit()
    return count

def training_selection(db,a,split='train'):
    print(f'Selecting {split} segments (training-style normal similarity and anomaly-aware deduplication)...',flush=True)
    # IDs in original partition order, followed by separate seeded shuffles.
    for label in (['unknown'] if a.annotation_mode=='unknown' else ['anomalous','normal']):
        ids=[r[0] for r in db.execute("SELECT id FROM segments WHERE split=? AND retained=1 AND label=? ORDER BY id",(split,label))]
        random.Random(a.seed).shuffle(ids)
        seen={};selected={};inverted=defaultdict(set);updates=[]
        for position,i in enumerate(ids,1):
            seq_json,signature=db.execute('SELECT sequence,anomaly_signature FROM segments WHERE id=?',(i,)).fetchone()
            seq=json.loads(seq_json)
            rep=frozenset(seq) if a.similarity_method=='jaccard' else Counter(seq)
            key=rep if a.similarity_method=='jaccard' else tuple(sorted(rep.items()))
            matched=None
            if label=='anomalous':
                key=(key,signature)
                matched=seen.get(key)
                if matched is None:seen[key]=i
                reason='unique_anomalous_pattern' if matched is None else 'duplicate_anomalous_pattern'
            else:
                candidates=set()
                for t in rep:candidates.update(inverted.get(t,()))
                # Disjoint representations have zero similarity; cannot reject
                # at any valid threshold. First threshold hit suffices for the
                # notebook's keep/drop decision; representative is not nearest.
                for j in sorted(candidates):
                    other=selected[j]
                    if a.similarity_method=='jaccard':score=len(rep&other)/len(rep|other)
                    else:
                        keys=rep.keys()|other.keys()
                        score=sum(min(rep[k],other[k]) for k in keys)/sum(max(rep[k],other[k]) for k in keys)
                    if score>=a.threshold:matched=j;break
                reason=f'{label}_below_threshold' if matched is None else f'{label}_similarity_at_or_above_threshold'
                if matched is None:
                    selected[i]=rep
                    for t in rep:inverted[t].add(i)
            updates.append((int(matched is None),reason,matched,i))
            if len(updates)>=10000:
                db.executemany('UPDATE segments SET retained=?,reason=?,representative=? WHERE id=?',updates)
                updates.clear();db.commit()
                print(f'{split} {label}: checked {position:,}/{len(ids):,}',flush=True)
        db.executemany('UPDATE segments SET retained=?,reason=?,representative=? WHERE id=?',updates)
        db.commit()
    counts=dict(db.execute("SELECT label,count(*) FROM segments WHERE split=? AND retained=1 GROUP BY label",(split,)))
    if set(counts)!=({'unknown'} if a.annotation_mode=='unknown' else {'normal','anomalous'}):
        raise ValueError(f'{split} selection must retain both classes; found {counts}.')

def process_evaluation(db):
    """Keep every test segment, including exact duplicates and reordered matches."""
    print('Keeping the complete test partition without deduplication...', flush=True)
    db.execute("UPDATE segments SET retained=1,reason='original_test_segment',representative=NULL WHERE split='test'")
    db.commit()


def exclude_validation_train_overlap(db):
    """Exclude exact ordered training matches after validation's first selection."""
    db.execute("UPDATE segments SET retained=0,reason='validation_sequence_seen_in_training' WHERE split='validation' AND retained=1 AND sequence IN (SELECT sequence FROM train_sequences)")
    db.commit()

# VALIDATION ONLY. All training-style selected points survive, except exact
# ordered-sequence overlaps with retained training, removed by the caller.
def record_validation_selection(db, a):
    """Record final validation selections without any representative subsampling."""
    raw = list(db.execute("SELECT id,label,sequence FROM segments WHERE split='validation' AND retained=1 ORDER BY id"))
    if not raw:
        raise ValueError('No validation candidates remain after training-style selection and train-overlap exclusion.')
    db.execute('CREATE TABLE validation_checkpoint_selection (id INTEGER PRIMARY KEY, selection_rank INTEGER NOT NULL)')
    db.executemany('INSERT INTO validation_checkpoint_selection(id,selection_rank) VALUES(?,?)',
                   ((sid,rank) for rank,(sid,_,_) in enumerate(raw,1)))
    db.commit()
    fields=['segment_id','selected','selection_rank','label','template_set_size']
    write_csv(a.output/'validation_selection_audit.csv',fields,
              (dict(segment_id=segment_id(sid),selected=1,selection_rank=rank,
                    label=label,template_set_size=len(set(json.loads(seq))))
               for rank,(sid,label,seq) in enumerate(raw,1)))
    counts=Counter(label for _,label,_ in raw)
    result=dict(method='training_style_only',
                validation_export='all_training_style_selected_validation_points',
                candidate_count=len(raw),selected_count=len(raw),removed_count=0,
                second_sampling_applied=False,
                class_counts_after=dict(counts),
                selected_segment_ids=[segment_id(sid) for sid,_,_ in raw],
                parameters=dict(similarity_method=a.similarity_method,threshold=a.threshold,seed=a.seed),
                note='No second-stage sampling or validation size cap. Exact ordered overlaps with retained training are excluded before this report.')
    (a.output/'validation_selection_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'Validation: {len(raw):,} segments retained by training-style selection and train-overlap exclusion; no second sampling.',flush=True)
    return result


EXPORT_FIELDS=['segment_id','split','status_gt_label','event_count','anomalous_event_count',
               'source_row_start','source_row_end','source_rows_json','Content','Content_with_labels',
               'messages_only','templates_json','event_labels_json','events_json','anomalous_templates_json']
ROW_QUERY='id,label,n,anomalies,first_row,last_row,sequence,payload,split,anomaly_signature'

def to_export(row):
    i,label,n,anom,first,last,seq,payload,split,signature=row
    return dict(segment_id=segment_id(i),split=split,status_gt_label=label,event_count=n,
                anomalous_event_count='' if label=='unknown' else anom,source_row_start=first,source_row_end=last,
                source_rows_json=dumps(list(range(first,last+1))),templates_json=seq,anomalous_templates_json='' if label=='unknown' else signature,**json.loads(payload))

def export_partitions(db,a):
    counts=Counter()
    @lru_cache(maxsize=100000)
    def category(seq):
        return 'seen' if db.execute('SELECT 1 FROM train_sequences WHERE sequence=?',(seq,)).fetchone() else 'unseen'
    train_anomaly_groups={anomaly_group(seq,sig,a.similarity_method) for seq,sig in db.execute("SELECT sequence,anomaly_signature FROM segments WHERE split='train' AND retained=1 AND label='anomalous'")}
    for split in SPLITS:
        path=partition_path(a,'train_unique' if split=='train' else split)
        with path.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=EXPORT_FIELDS+(['checkpoint_selected'] if split=='validation' else [])+(['category','anomalous_pattern_category'] if split=='test' else []));writer.writeheader()
            for raw in db.execute(f'SELECT {ROW_QUERY} FROM segments WHERE split=? AND retained=1 ORDER BY id',(split,)):
                row=to_export(raw)
                if split=='validation':
                    row['checkpoint_selected']=int(db.execute(
                        'SELECT 1 FROM validation_checkpoint_selection WHERE id=?',(raw[0],)).fetchone() is not None)
                if split=='test':
                    row['category']=category(raw[6])
                    row['anomalous_pattern_category']=('seen' if anomaly_group(raw[6],raw[9],a.similarity_method) in train_anomaly_groups else 'unseen') if row['status_gt_label']=='anomalous' else 'not_applicable'
                    counts[row['category'],row['status_gt_label']]+=1
                writer.writerow(row)
    fields=['segment_id','split','status_gt_label','source_row_start','source_row_end','retained_in_export','reason','representative_segment_id']
    with (a.output/'split_assignment.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(fields)
        for i,split,label,first,last,retained,reason,rep in db.execute('SELECT id,split,label,first_row,last_row,retained,reason,representative FROM segments ORDER BY id'):
            writer.writerow([segment_id(i),split,label,first,last,retained,reason,segment_id(rep)])
    return counts

def oversample(db,a):
    if a.annotation_mode=='unknown':
        n=db.execute("SELECT count(*) FROM segments WHERE split='train' AND retained=1").fetchone()[0]
        return dict(applicable=False,reason='unannotated_dataset',unique_unknown=n,total_training_examples=n,added_copies=0)
    counts=dict(db.execute("SELECT label,count(*) FROM segments WHERE split='train' AND retained=1 GROUP BY label"))
    target=counts['anomalous'] if a.no_oversampling else max(counts['anomalous'],math.ceil(counts['normal']*a.anomaly_to_normal_ratio))
    extra=target-counts['anomalous'];example=0
    with partition_path(a,'train_oversampled').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=EXPORT_FIELDS+['training_example_id','is_oversampled']);writer.writeheader()
        def emit(raw,is_copy):
            nonlocal example
            example+=1;row=to_export(raw)
            row.update(training_example_id=f'train_example_{example:09d}',is_oversampled=int(is_copy));writer.writerow(row)
        for raw in db.execute(f"SELECT {ROW_QUERY} FROM segments WHERE split='train' AND retained=1 ORDER BY id"):emit(raw,False)
        # Evenly repeat selected anomalous examples; shuffle each extra pass.
        ids=[r[0] for r in db.execute("SELECT id FROM segments WHERE split='train' AND label='anomalous' AND retained=1 ORDER BY id")]
        rng=random.Random(a.seed);remaining=extra
        while remaining:
            rng.shuffle(ids)
            for i in ids[:remaining]:
                emit(db.execute(f'SELECT {ROW_QUERY} FROM segments WHERE id=?',(i,)).fetchone(),True)
                remaining-=1
    return dict(unique_normal=counts['normal'],unique_anomalous=counts['anomalous'],
                added_anomalous_copies=extra,final_normal=counts['normal'],final_anomalous=target,
                total_training_examples=example,achieved_anomaly_to_normal_ratio=target/counts['normal'])

def report(db,a,input_events,dropped,conflicts,cats,oversampling,elapsed):
    partitions=[];removals=[];warnings=[]
    for split in SPLITS:
        before=dict(db.execute('SELECT label,count(*) FROM segments WHERE split=? GROUP BY label',(split,)))
        after=dict(db.execute('SELECT label,count(*) FROM segments WHERE split=? AND retained=1 GROUP BY label',(split,)))
        for label in (['unknown','all'] if a.annotation_mode=='unknown' else ['normal','anomalous','all']):
            n=sum(before.values()) if label=='all' else before.get(label,0)
            k=sum(after.values()) if label=='all' else after.get(label,0)
            partitions.append(dict(partition=split,label=label,initial_segments=n,retained_unique_segments=k,removed_segments=n-k))
        removals.extend(dict(partition=split,label=l,reason=r,removed_segments=n) for l,r,n in db.execute('SELECT label,reason,count(*) FROM segments WHERE split=? AND retained=0 GROUP BY label,reason',(split,)))
        if a.annotation_mode=='annotated' and split!='train' and (after.get('normal',0)==0 or after.get('anomalous',0)==0):
            warnings.append(f'{split} lacks one or both classes after filtering: {after}. Inspect counts before training; this partition cannot support a two-class evaluation as-is.')
        if split!='train' and not sum(after.values()):
            warnings.append(f'{split} is empty after filtering; evaluation requires retained segments.')
    total=db.execute('SELECT count(*) FROM segments').fetchone()[0]
    kept=db.execute('SELECT count(*) FROM segments WHERE retained=1').fetchone()[0]
    assert sum(r['initial_segments'] for r in partitions if r['label']=='all')==total
    assert sum(r['retained_unique_segments'] for r in partitions if r['label']=='all')==kept
    assert total-kept==sum(r['removed_segments'] for r in removals)
    assert db.execute("SELECT count(*) FROM segments WHERE split='validation' AND retained=1 AND sequence IN (SELECT sequence FROM train_sequences)").fetchone()[0]==0
    # Repeated sequences are explicitly allowed in the untouched test partition.
    n,distinct=db.execute("SELECT count(*),count(DISTINCT json_array(sequence,anomaly_signature)) FROM segments WHERE split='validation' AND retained=1").fetchone()
    assert n==distinct
    assert db.execute("SELECT count(*) FROM segments WHERE split='test' AND retained=1").fetchone()[0] == db.execute("SELECT count(*) FROM segments WHERE split='test'").fetchone()[0]
    assert db.execute('SELECT sum(n) FROM segments').fetchone()[0]+dropped==input_events
    catrows=[dict(category=c,normal=cats[c,'normal'],anomalous=cats[c,'anomalous'],total=cats[c,'normal']+cats[c,'anomalous']) for c in CATEGORIES]
    if a.annotation_mode=='unknown':
        catrows=[dict(category=c,unknown=cats[c,'unknown'],total=cats[c,'unknown']) for c in CATEGORIES]
    assert sum(r['total'] for r in catrows)==db.execute("SELECT count(*) FROM segments WHERE split='test' AND retained=1").fetchone()[0]
    if a.annotation_mode=='annotated' and any(r['anomalous']==0 for r in catrows):warnings.append('At least one test seen/unseen category has no anomalous segments; report this limitation rather than an anomaly recall for that subset.')
    if a.line_id_gaps:warnings.append(f'Explicitly allowed {a.line_id_gaps} source LineId discontinuities.')
    write_csv(a.output/'partition_counts.csv',list(partitions[0]),partitions)
    write_csv(a.output/'removal_counts.csv',['partition','label','reason','removed_segments'],removals)
    write_csv(a.output/'test_category_counts.csv',list(catrows[0]),catrows)
    notes=['No global deduplication before splitting.',
           'Normal training uses greedy threshold similarity; anomalous training uses set/count plus anomalous-template-subset deduplication.',
           'Validation uses the same selection rules as training, then excludes exact retained training sequences. Every remaining point is exported and checkpoint flagged. There is no second sampling or size limit.',
           'Test preserves all original segments (including duplicates); seen/unseen compares against final unique training.',
           'Oversampling changes exposures only, not seen/unseen categories.',
           'Configured split ratios apply before filtering; final counts may differ.',
           'Test retains every originally assigned segment, including repeated patterns; evaluation reflects their original frequencies.',
           'Recompute overlap filtering/categories if training selection changes.',
           'Content_with_labels and other ground-truth columns must not enter test inference.']
    if a.annotation_mode=='unknown':
        notes[1]='Unknown training uses greedy threshold similarity across all candidates.'
        notes[2]='Validation uses training-style similarity filtering and excludes retained training ordered sequences. All remaining points are exported without further sampling. Test retains every originally assigned segment, including duplicates.'
        notes[4]='No oversampling for unannotated data. Anomaly fields are not applicable.'
        if not a.timestamp_order_checked:
            notes.append('Timestamp order was not checked; input CSV row order was preserved.')
    result=dict(annotation_mode=a.annotation_mode,timestamp_order_checked=a.timestamp_order_checked,status='completed',input_file=str(a.input),input_events=input_events,window_size=a.window_size,
                trailing_events_dropped=dropped,segments_created=total,unique_segments_retained=kept,
                segments_removed=total-kept,conflicting_sequence_groups=conflicts,
                partitions=partitions,removals=removals,oversampling=oversampling,test_categories=catrows,
                warnings=warnings,notes=notes,elapsed_seconds=round(elapsed,2))
    (a.output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=[f'Input events: {input_events:,}',f'Window size: {a.window_size}',f'Split ratios: {a.ratios}',
           f'Segments created: {total:,}',f'Unique segments retained: {kept:,}',f'Segments removed: {total-kept:,}',
           f'Trailing events dropped: {dropped:,}',f'Conflicting sequence groups: {conflicts}', '', 'PARTITION COUNTS (before oversampling)']
    lines += [json.dumps(r) for r in partitions]+['','TRAINING OVERSAMPLING',json.dumps(oversampling),'','TEST CATEGORIES']
    lines += [json.dumps(r) for r in catrows]+['','WARNINGS']+warnings+['','NOTES']+notes
    text='\n'.join(lines)+'\n';(a.output/'report.txt').write_text(text,encoding='utf-8');print(text)

def main():
    a=arguments();csv.field_size_limit(min(sys.maxsize,2147483647));a.output.mkdir(parents=True,exist_ok=True)
    if a.analyze_existing is not None:
        ref=a.analyze_existing
        analyze_novelty(*existing_partitions(ref),a.output)
        print(f'Enriched test.csv and novelty reports: {a.output}')
        return
    work=a.output if a.work_dir is None else a.work_dir.resolve()
    work.mkdir(parents=True,exist_ok=True)
    os.environ['SQLITE_TMPDIR']=str(work)
    fd,name=tempfile.mkstemp(prefix='log_segments_',suffix='.sqlite',dir=work);os.close(fd);db_path=Path(name)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
    config['partition_files']={role:partition_path(a,role).name for role in ['train_unique','train_oversampled','validation','test']}
    config.update(version='2.0-proxifier-full-test',database_path=str(db_path),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  split_method='per_label_seeded_rank_largest_remainder',training_shuffle='independent random.Random(seed) per class',
                  source_order='input_csv_row_order',category_reference='final unique training')
    (a.output/'configuration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    db=sqlite3.connect(db_path);started=time.monotonic()
    try:
        setup(db)
        input_events,dropped,headers=ingest(db,a)
        config.update(annotation_mode=a.annotation_mode, input_has_label_column=a.input_has_label, timestamp_order_checked=a.timestamp_order_checked,
                      header_columns_used=headers, split_method='seeded_rank_largest_remainder' if a.annotation_mode=='unknown' else 'per_label_seeded_rank_largest_remainder',
                      oversampling_applicable=a.annotation_mode=='annotated')
        (a.output/'configuration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        if a.annotation_mode=='unknown':
            config['partition_files'].pop('train_oversampled',None)
            (a.output/'configuration.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        split_segments(db,a)
        conflicts=detect_conflicts(db,a)
        training_selection(db,a)
        build_reference(db)
        training_selection(db,a,split='validation')
        exclude_validation_train_overlap(db)
        process_evaluation(db)
        selection=record_validation_selection(db,a)
        print('Exporting partitions...',flush=True)
        cats=export_partitions(db,a)
        oversampling=oversample(db,a)
        report(db,a,input_events,dropped,conflicts,cats,oversampling,time.monotonic()-started)
        print('Analyzing test sequence and individual-template novelty...',flush=True)
        novelty=analyze_novelty(partition_path(a,'train_unique'),partition_path(a,'validation'),partition_path(a,'test'),a.output)
        coverage=training_coverage(db,a)
        report_path=a.output/'report.json'
        full_report=json.loads(report_path.read_text(encoding='utf-8'));full_report['test_novelty']=novelty;full_report['training_coverage']=coverage;full_report['validation_selection']=selection
        report_path.write_text(json.dumps(full_report,indent=2),encoding='utf-8')
        with (a.output/'report.txt').open('a',encoding='utf-8') as f:
            f.write('\n'+(a.output/'novelty_report.txt').read_text(encoding='utf-8'))
            f.write('\nTRAINING COVERAGE\n'+json.dumps(coverage,indent=2)+'\n')
            f.write('\nVALIDATION SELECTION\n'+json.dumps(selection,indent=2)+'\n')
    except Exception as exc:
        failure={'status':'failed','error':str(exc),'database_path':str(db_path)}
        try:(a.output/'failure_report.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
        except OSError:print(json.dumps(failure),file=sys.stderr)
        raise
    finally:db.close()
    if not a.keep_work_db:db_path.unlink()
    print(f'Outputs: {a.output}',flush=True)


# Additional diagnostic comparisons. These NEVER change the split, retention,
# oversampling or original category=seen/unseen (exact TRAINING sequence match).
NOVELTY_RELATIONS = ['exact_sequence', 'same_counts_different_order',
                     'same_template_set_different_counts',
                     'new_combination_known_templates', 'contains_unseen_templates']

def analyze_novelty(train_path, validation_path, test_path, output_dir):
    """Stream existing partitions; enrich test.csv and export coverage reports.

    'train_or_validation' matches one complete segment in either partition;
    it never stitches two segments together. 'Known templates' may occur across
    multiple reference segments. Template counts are DISTINCT per test segment.
    All reference files must be final unique train and filtered validation.
    """
    output_dir=Path(output_dir)
    fd,name=tempfile.mkstemp(prefix='novelty_',suffix='.sqlite',dir=output_dir);os.close(fd)
    con=sqlite3.connect(name)
    con.executescript('''
      PRAGMA temp_store=FILE;
      CREATE TABLE patterns(ref TEXT,kind TEXT,key TEXT,representative TEXT,
        PRIMARY KEY(ref,kind,key)) WITHOUT ROWID;
      CREATE TABLE templates(template TEXT PRIMARY KEY,train INTEGER DEFAULT 0,
        validation INTEGER DEFAULT 0,test_segments INTEGER DEFAULT 0) WITHOUT ROWID;
    ''')
    def sequence(row,path):
        try:seq=json.loads(row['templates_json'])
        except (KeyError,ValueError) as exc:raise ValueError(f'{path}: missing/invalid templates_json') from exc
        if not isinstance(seq,list) or not seq or any(not isinstance(t,str) or not t.strip() for t in seq):
            raise ValueError(f'{path}: templates_json must be a nonempty list of nonblank strings.')
        return seq
    def keys(seq):
        return [('sequence',dumps(seq)),('counts',dumps(sorted(Counter(seq).items()))),('set',dumps(sorted(set(seq))))]
    try:
        for ref,path in [('train',train_path),('validation',validation_path)]:
            with Path(path).open(encoding='utf-8-sig',newline='') as f:
                reader=csv.DictReader(f)
                if not {'segment_id','templates_json'}<=set(reader.fieldnames or []):raise ValueError(f'{path}: segment_id and templates_json required.')
                for number,row in enumerate(reader,1):
                    seq=sequence(row,path)
                    con.executemany('INSERT OR IGNORE INTO patterns VALUES(?,?,?,?)',[(ref,k,v,row['segment_id']) for k,v in keys(seq)])
                    for t in set(seq):
                        con.execute('INSERT OR IGNORE INTO templates(template) VALUES(?)',(t,))
                        con.execute(f'UPDATE templates SET {ref}=1 WHERE template=?',(t,))
                    if number%10000==0:con.commit()
            con.commit()
        @lru_cache(maxsize=100000)
        def membership(t):
            row=con.execute('SELECT train,validation FROM templates WHERE template=?',(t,)).fetchone()
            return (False,False) if row is None else (bool(row[0]),bool(row[1]))
        def match(seq,refs):
            for kind,key in keys(seq):
                for ref in refs:
                    found=con.execute('SELECT representative FROM patterns WHERE ref=? AND kind=? AND key=?',(ref,kind,key)).fetchone()
                    if found:
                        rel={'sequence':'exact_sequence','counts':'same_counts_different_order','set':'same_template_set_different_counts'}[kind]
                        return rel,ref,found[0]
            known=all(any(membership(t)[0 if ref=='train' else 1] for ref in refs) for t in set(seq))
            return ('new_combination_known_templates' if known else 'contains_unseen_templates'),'',''
        extra=[]
        for ref in ['train','validation','train_or_validation']:
            extra += [f'relation_to_{ref}',f'matching_{ref}_segment_id',f'matching_{ref}_partition']
        extra += ['distinct_template_count','templates_seen_in_train_count','templates_seen_in_validation_count',
                  'templates_seen_in_either_count','templates_unseen_in_train_count','templates_unseen_in_validation_count',
                  'templates_unseen_in_both_count','templates_unseen_in_train_json',
                  'templates_unseen_in_validation_json','templates_unseen_in_both_json']
        temp=output_dir/'test_enriched.csv.partial'
        totals=Counter();by_label=Counter();original_unseen=Counter();n=0
        with Path(test_path).open(encoding='utf-8-sig',newline='') as f,temp.open('x',encoding='utf-8',newline='') as dst:
            reader=csv.DictReader(f);fields=reader.fieldnames or []
            if not {'segment_id','templates_json','status_gt_label','category'}<=set(fields):
                raise ValueError('Test CSV needs segment_id, templates_json, status_gt_label and category.')
            writer=csv.DictWriter(dst,fieldnames=[k for k in fields if k not in extra]+extra);writer.writeheader()
            for row in reader:
                seq=sequence(row,test_path);templates=sorted(set(seq));n+=1
                for ref,refs in [('train',['train']),('validation',['validation']),('train_or_validation',['train','validation'])]:
                    relation,matched_ref,sid=match(seq,refs)
                    row[f'relation_to_{ref}']=relation
                    row[f'matching_{ref}_segment_id']=sid;row[f'matching_{ref}_partition']=matched_ref
                    totals[ref,relation]+=1;by_label[ref,relation,row['status_gt_label']]+=1
                    if row['category']=='unseen':original_unseen[ref,relation]+=1
                expected='seen' if row['relation_to_train']=='exact_sequence' else 'unseen'
                if row['category']!=expected:
                    raise ValueError(f"Stale category for {row['segment_id']}: use train_unique.csv from the SAME run. Expected {expected}.")
                unseen_train=[t for t in templates if not membership(t)[0]]
                unseen_val=[t for t in templates if not membership(t)[1]]
                unseen_both=[t for t in templates if not any(membership(t))]
                row.update(distinct_template_count=len(templates),templates_seen_in_train_count=len(templates)-len(unseen_train),
                           templates_seen_in_validation_count=len(templates)-len(unseen_val),
                           templates_seen_in_either_count=len(templates)-len(unseen_both),
                           templates_unseen_in_train_count=len(unseen_train),templates_unseen_in_validation_count=len(unseen_val),
                           templates_unseen_in_both_count=len(unseen_both),templates_unseen_in_train_json=dumps(unseen_train),
                           templates_unseen_in_validation_json=dumps(unseen_val),templates_unseen_in_both_json=dumps(unseen_both))
                writer.writerow(row)
                for t in templates:
                    con.execute('INSERT OR IGNORE INTO templates(template) VALUES(?)',(t,))
                    con.execute('UPDATE templates SET test_segments=test_segments+1 WHERE template=?',(t,))
                if n%10000==0:con.commit()
        con.commit()
        # Input has been closed before replacement (also works on Windows).
        temp.replace(output_dir/Path(test_path).name)
        coverage=Counter()
        with (output_dir/'test_template_coverage.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.writer(f);writer.writerow(['EventTemplate','seen_in_train','seen_in_validation','coverage_category','test_segment_count'])
            for t,tr,va,count in con.execute('SELECT * FROM templates WHERE test_segments>0 ORDER BY template'):
                category='seen_in_both' if tr and va else 'train_only' if tr else 'validation_only' if va else 'unseen_in_both'
                coverage[category]+=1;writer.writerow([t,tr,va,category,count])
        rows=[]
        for ref in ['train','validation','train_or_validation']:
            for rel in NOVELTY_RELATIONS:
                rows.append(dict(reference=ref,relation=rel,test_segments=totals[ref,rel],
                                 normal=by_label[ref,rel,'normal'],anomalous=by_label[ref,rel,'anomalous'],
                                 originally_unseen_in_train=original_unseen[ref,rel]))
        if any(k[2]=='unknown' for k in by_label):
            for row in rows:
                row['unknown']=by_label[row['reference'],row['relation'],'unknown']
        write_csv(output_dir/'test_novelty_counts.csv',list(rows[0]),rows)
        template_summary={key:coverage[key] for key in ['seen_in_both','train_only','validation_only','unseen_in_both']}
        template_summary.update(total_distinct_test_templates=sum(coverage.values()),
            distinct_test_templates_seen_in_train=coverage['seen_in_both']+coverage['train_only'],
            distinct_test_templates_seen_in_validation=coverage['seen_in_both']+coverage['validation_only'],
            distinct_test_templates_seen_in_either=sum(coverage.values())-coverage['unseen_in_both'])
        summary=dict(test_segments=n,segment_comparisons=rows,distinct_template_coverage=template_summary,
                     notes=['Original category remains exact-sequence familiarity to TRAINING only.',
                            'Same counts/different order means reordering alone explains the difference.',
                            'Same template set/different counts means repetitions or segment length differ too.',
                            'Known templates may have occurred across different reference segments.',
                            'Either-partition pattern match must match ONE complete train or validation segment.',
                            'Validation familiarity is not training exposure.',
                            'Per-segment template counts are DISTINCT; do not sum them to obtain global counts.'])
        (output_dir/'novelty_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        lines=['TEST SEQUENCE NOVELTY (counts are mutually exclusive within each reference)']+[json.dumps(r) for r in rows]
        lines+=['','DISTINCT TEST TEMPLATE COVERAGE',json.dumps(template_summary),'']+summary['notes']
        (output_dir/'novelty_report.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        return summary
    finally:
        con.close();Path(name).unlink(missing_ok=True)


def anomaly_group(seq_json,signature,method):
    seq=json.loads(seq_json)
    representation=tuple(sorted(set(seq))) if method=='jaccard' else tuple(sorted(Counter(seq).items()))
    return representation,tuple(json.loads(signature))

def training_coverage(db,a):
    all_before=set();all_after=set();anom_before=set();anom_after=set()
    groups_before=set();groups_after=set()
    for seq,sig,label,retained in db.execute("SELECT sequence,anomaly_signature,label,retained FROM segments WHERE split='train'"):
        ts=set(json.loads(seq));ats=set(json.loads(sig))
        all_before.update(ts);anom_before.update(ats)
        if retained:all_after.update(ts);anom_after.update(ats)
        if label=='anomalous':
            key=anomaly_group(seq,sig,a.similarity_method);groups_before.add(key)
            if retained:groups_after.add(key)
    rows=[]
    for t in sorted(all_before):
        rows.append(dict(EventTemplate=t,in_training_before=1,in_training_after=int(t in all_after),
                         anomalous_event_before=int(t in anom_before),anomalous_event_after=int(t in anom_after)))
    write_csv(a.output/'training_template_coverage.csv',['EventTemplate','in_training_before','in_training_after','anomalous_event_before','anomalous_event_after'],rows)
    result=dict(all_templates_before=len(all_before),all_templates_after=len(all_after),
                missing_templates=sorted(all_before-all_after),anomalous_event_templates_before=len(anom_before),
                anomalous_event_templates_after=len(anom_after),missing_anomalous_event_templates=sorted(anom_before-anom_after),
                anomalous_groups_before=len(groups_before),anomalous_groups_after=len(groups_after),
                anomalous_groups_lost=len(groups_before-groups_after),timestamp_order_checked=a.timestamp_order_checked,
                coverage_reference='initial training partition vs retained training before oversampling',partitions=[])
    for split in ['validation','test']:
        groups=set();counts=Counter()
        for seq,sig in db.execute("SELECT sequence,anomaly_signature FROM segments WHERE split=? AND retained=1 AND label='anomalous'",(split,)):
            key=anomaly_group(seq,sig,a.similarity_method);groups.add(key)
            counts['seen' if key in groups_after else 'unseen']+=1
        result['partitions'].append(dict(partition=split,anomalous_segments_seen_pattern=counts['seen'],
             anomalous_segments_unseen_pattern=counts['unseen'],distinct_anomalous_groups=len(groups),
             distinct_groups_seen_in_train=len(groups&groups_after),distinct_groups_unseen_in_train=len(groups-groups_after)))
    if a.annotation_mode=='unknown':
        result = {k:v for k,v in result.items() if k in ['all_templates_before','all_templates_after','missing_templates','timestamp_order_checked','coverage_reference']}
        result.update(annotation_mode='unknown', anomaly_coverage_applicable=False)
        write_csv(a.output/'training_template_coverage.csv',['EventTemplate','in_training_before','in_training_after'],
                  ({k:r[k] for k in ['EventTemplate','in_training_before','in_training_after']} for r in rows))
    (a.output/'training_coverage_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,sqlite3.Error,csv.Error) as exc:
        print(f'ERROR: {exc}\nPartial outputs are not completed partitions. Use a new output directory when rerunning.',file=sys.stderr)
        sys.exit(1)
