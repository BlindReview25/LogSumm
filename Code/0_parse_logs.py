#!/usr/bin/env python3
"""Stream BGL, Apache, Linux, OpenStack and ZooKeeper logs to separate CSVs.

Python 3.9+, standard library only. Run in the directory containing logs:
  python parse_unlabeled_logs.py
  python parse_unlabeled_logs.py --input Apache_full.log Linux_full.log
  python parse_unlabeled_logs.py --dataset linux --input custom_name.log
  python parse_unlabeled_logs.py --input "*.log" --output-dir parsed

With no inputs, discovers recognized dataset names among *.log files in the
current directory. --dataset selects a parser for all explicitly supplied files.
Otherwise the parser is inferred from each filename (case insensitive).

One physical line becomes one CSV row. LineId is the physical line number.
No sorting, deduplication, segmentation, template mining or inferred anomaly labels.
BGL retains original event labels and omits NodeRepeat. Other datasets have no Label column.
Content preserves message whitespace after one header delimiter character.
EventId and EventTemplate are blank for subsequent template extraction.
Linux has no year/timezone: no date or epoch timestamp is fabricated. Other
recorded date/time values are also left unchanged, with no timezone inference.
OpenStack request context is retained in RequestContext, without brackets;
instance details remain in Content. ZooKeeper Location is retained verbatim,
with Thread, Component and SourceLine also extracted when recognizable.

Default: malformed lines fail that file and leave a .partial CSV and report.
--on-error preserve keeps them as Content, with ParseStatus=unparsed, and logs
errors separately. It never silently drops lines or guesses continuation headers.
Resolve those rows before downstream experiments. --include-raw stores every
original line in RawLog (line endings excluded); it increases output size.
Existing outputs are never overwritten. Batch processing continues after a
file failure and returns a nonzero exit code if any file failed.
"""
import argparse
import csv
import glob
import json
import re
import sys
import time
from pathlib import Path

APACHE = re.compile(r'^\[(?P<Time>[^\]]+)\][ \t]+\[(?P<Level>[^\]]+)\](?:[ \t](?P<Content>.*))?$')
LINUX = re.compile(r'^(?P<Month>[A-Za-z]{3})[ \t]+(?P<Day>\d{1,2})[ \t]+(?P<Time>\d{2}:\d{2}:\d{2})[ \t]+(?P<Host>\S+)[ \t]+(?P<Component>[^:]+):(?:[ \t]?(?P<Content>.*))?$')
OPENSTACK = re.compile(r'^(?:(?P<LogFile>\S+)[ \t]+)?(?P<Date>\d{4}-\d{2}-\d{2})[ \t]+(?P<Time>\d{2}:\d{2}:\d{2}[.,]\d+)[ \t]+(?P<PID>\d+)[ \t]+(?P<Level>\S+)[ \t]+(?P<Component>\S+)(?:[ \t](?P<Content>.*))?$')
ZOOKEEPER = re.compile(r'^(?P<Date>\d{4}-\d{2}-\d{2})[ \t]+(?P<Time>\d{2}:\d{2}:\d{2}[.,]\d+)[ \t]+-[ \t]+(?P<Level>\S+)[ \t]+\[(?P<Location>.*?)\][ \t]+-(?:[ \t]?(?P<Content>.*))?$')


def match_fields(pattern, raw, dataset):
    m = pattern.fullmatch(raw)
    if not m:
        raise ValueError('Unrecognized ' + dataset + ' header')
    return {k: v if v is not None else '' for k, v in m.groupdict().items()}


def parse_apache(raw):
    return match_fields(APACHE, raw, 'Apache')


def parse_linux(raw):
    row = match_fields(LINUX, raw, 'Linux')
    if row['Month'] not in 'Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split() or not 1 <= int(row['Day']) <= 31:
        raise ValueError('Invalid Linux month/day')
    row['PID'] = ''
    m = re.fullmatch(r'(.*)\[(\d+)\]', row['Component'])
    if m:
        row['Component'], row['PID'] = m.groups()
    return row


def parse_openstack(raw):
    row = match_fields(OPENSTACK, raw, 'OpenStack')
    # Only the first request-context block is metadata; do not swallow
    # subsequent [instance: ...] blocks belonging to the message.
    row['RequestContext'] = ''
    m = re.match(r'^\[(?P<context>(?:req-[^\]]*|-))\](?:[ \t]|$)', row['Content'])
    if m:
        row['RequestContext'] = m.group('context')
        row['Content'] = row['Content'][m.end():]
    return row


def parse_zookeeper(raw):
    row = match_fields(ZOOKEEPER, raw, 'ZooKeeper')
    row.update(Thread='', Component='', SourceLine='')
    # Greedy Thread handles thread names that themselves contain colons.
    m = re.fullmatch(r'(.*):([^:@]+)@(\d+)', row['Location'])
    if m:
        row['Thread'], row['Component'], row['SourceLine'] = m.groups()
    return row


BGL_FIELDS = ['Label', 'Timestamp', 'Date', 'Node', 'Time', 'NodeRepeat',
              'Type', 'Component', 'Level']
BGL = re.compile(r'^' + r'[ \t]+'.join(
    f'(?P<{name}>\\S+)' for name in BGL_FIELDS
) + r'(?:[ \t](?P<Content>.*))?$')


def parse_bgl(raw):
    row = match_fields(BGL, raw, 'BGL')
    if not re.fullmatch(r'\d+', row['Timestamp']):
        raise ValueError('BGL Timestamp is not an unsigned integer')
    if not re.fullmatch(r'\d{4}\.\d{2}\.\d{2}', row['Date']):
        raise ValueError('BGL Date must have YYYY.MM.DD format')
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}(?:\.\d+)?', row['Time']):
        raise ValueError('Unrecognized BGL Time format')
    row.pop('NodeRepeat')
    return row


PARSERS = {'bgl': parse_bgl, 'apache': parse_apache, 'linux': parse_linux,
           'openstack': parse_openstack, 'zookeeper': parse_zookeeper}
FIELDS = {
    'bgl': ['Label', 'Timestamp', 'Date', 'Node', 'Time', 'Type', 'Component', 'Level', 'Content'],
    'apache': ['Time', 'Level', 'Content'],
    'linux': ['Month', 'Day', 'Time', 'Host', 'Component', 'PID', 'Content'],
    'openstack': ['LogFile', 'Date', 'Time', 'PID', 'Level', 'Component', 'RequestContext', 'Content'],
    'zookeeper': ['Date', 'Time', 'Level', 'Location', 'Thread', 'Component', 'SourceLine', 'Content'],
}


def infer_dataset(path):
    matches = [name for name in PARSERS if name in path.name.lower()]
    if len(matches) != 1:
        raise ValueError('Cannot infer one dataset from ' + path.name + '; use --dataset.')
    return matches[0]


def convert(source, dataset, args):
    output_dir = args.output_dir.resolve() if args.output_dir else source.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / (source.stem + '_structured.csv')
    partial = target.with_name(target.name + '.partial')
    report = target.with_name(target.stem + '_parse_report.json')
    errors = target.with_name(target.stem + '_parse_errors.csv')
    for path in (target, partial, report, errors):
        if path.exists():
            raise ValueError('Refusing to overwrite ' + str(path) + '; choose a new --output-dir.')
    fields = ['LineId', 'SourceFile'] + FIELDS[dataset] + ['EventId', 'EventTemplate', 'ParseStatus']
    if args.include_raw:
        fields += ['RawLog']
    stats = dict(dataset=dataset, input=str(source), output=str(target), status='running',
                 lines_read=0, rows_written=0, parsed_rows=0, unparsed_rows=0,
                 columns=fields, encoding=args.encoding, on_error=args.on_error,
                 row_order='source_file_order', labels_inferred=False, templates_extracted=False)
    if dataset == 'bgl':
        stats.update(normal_rows=0, anomalous_rows=0)
    started = time.monotonic()
    failure = None
    try:
        with source.open(encoding=args.encoding, errors='strict', newline='') as src, \
             partial.open('x', encoding='utf-8', newline='') as dst, \
             errors.open('x', encoding='utf-8', newline='') as err:
            writer = csv.DictWriter(dst, fieldnames=fields)
            writer.writeheader()
            ew = csv.writer(err)
            ew.writerow(['LineId', 'Error', 'RawLog'])
            for number, line in enumerate(src, 1):
                stats['lines_read'] = number
                raw = line[:-2] if line.endswith('\r\n') else line[:-1] if line.endswith(('\r', '\n')) else line
                try:
                    row = PARSERS[dataset](raw)
                except ValueError as exc:
                    stats['unparsed_rows'] += 1
                    ew.writerow([number, str(exc), raw])
                    if args.on_error == 'fail':
                        raise ValueError(f'Line {number}: {exc}; see {errors.name}') from exc
                    row = dict(Content=raw, ParseStatus='unparsed')
                else:
                    row['ParseStatus'] = 'parsed'
                    stats['parsed_rows'] += 1
                    if dataset == 'bgl':
                        stats['normal_rows' if row['Label'] == '-' else 'anomalous_rows'] += 1
                row.update(LineId=number, SourceFile=source.name, EventId='', EventTemplate='')
                if args.include_raw:
                    row['RawLog'] = raw
                writer.writerow(row)
                stats['rows_written'] += 1
                if args.progress_every and number % args.progress_every == 0:
                    print(f'{source.name}: {number:,} lines', flush=True)
        if not stats['lines_read']:
            raise ValueError('Input file is empty')
        partial.rename(target)
        stats['status'] = 'completed_with_unparsed_rows' if stats['unparsed_rows'] else 'completed'
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        failure = str(exc)
        stats.update(status='failed', error=failure, partial_output=str(partial))
    stats['elapsed_seconds'] = round(time.monotonic() - started, 2)
    try:
        with report.open('x', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
            f.write('\n')
    except OSError as exc:
        print(f'Cannot save report {report}: {exc}', file=sys.stderr)
        return False
    if failure:
        print(f'FAILED {source.name}: {failure}', file=sys.stderr)
        return False
    print(f'Saved {target}: {stats["parsed_rows"]:,} parsed, {stats["unparsed_rows"]:,} unparsed')
    if stats['unparsed_rows']:
        print('Resolve rows with ParseStatus=unparsed before segmentation.')
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', nargs='+', help='One or more paths or glob patterns; default: recognized *.log files in current directory')
    p.add_argument('--dataset', type=str.lower, choices=['auto'] + list(PARSERS), default='auto')
    p.add_argument('--output-dir', type=Path, help='Default: beside each input')
    p.add_argument('--encoding', default='utf-8-sig')
    p.add_argument('--on-error', choices=['fail', 'preserve'], default='fail')
    p.add_argument('--include-raw', action='store_true')
    p.add_argument('--progress-every', type=int, default=100000)
    a = p.parse_args()
    if a.progress_every < 0:
        p.error('--progress-every must be nonnegative')
    paths = []
    if a.input:
        for item in a.input:
            found = [item] if Path(item).is_file() else sorted(glob.glob(item))
            if not found:
                p.error('No input matches: ' + item)
            paths.extend(Path(x).resolve() for x in found)
    else:
        for path in sorted(Path.cwd().iterdir()):
            if not path.is_file() or path.suffix.lower() != '.log':
                continue
            try:
                dataset = infer_dataset(path)
            except ValueError:
                continue
            if a.dataset == 'auto' or dataset == a.dataset:
                paths.append(path.resolve())
    paths = list(dict.fromkeys(paths))
    if not paths:
        p.error('No matching log files. Supply --input and, for custom names, --dataset.')
    jobs = []
    targets = set()
    for source in paths:
        if not source.is_file():
            p.error('Not a file: ' + str(source))
        try:
            dataset = infer_dataset(source) if a.dataset == 'auto' else a.dataset
        except ValueError as exc:
            p.error(str(exc))
        target = (a.output_dir.resolve() if a.output_dir else source.parent) / (source.stem + '_structured.csv')
        if target in targets:
            p.error('Output filename collision: ' + str(target))
        targets.add(target)
        jobs.append((source, dataset))
    success = True
    for source, dataset in jobs:
        try:
            ok = convert(source, dataset, a)
        except (OSError, ValueError) as exc:
            print(f'FAILED {source.name}: {exc}', file=sys.stderr)
            ok = False
        success = ok and success
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
