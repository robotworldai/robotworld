"""Summarize canonical asset directories without following links or exposing files."""
import json
import os
from pathlib import Path

WORLD = Path(__file__).resolve().parents[2]


def main():
    root = WORLD/'Assets'
    rows = []
    global_inodes = set()
    global_bytes = 0
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and not p.is_symlink()):
        files = size = unique_size = links = 0
        inodes = set()
        for base, dirs, names in os.walk(directory, followlinks=False):
            for name in dirs + names:
                path = Path(base)/name
                if path.is_symlink():
                    links += 1
                    continue
                if not path.is_file():continue
                stat = path.stat()
                key = (stat.st_dev, stat.st_ino)
                files += 1
                size += stat.st_size
                if key not in inodes:
                    unique_size += stat.st_size
                    inodes.add(key)
                if key not in global_inodes:
                    global_bytes += stat.st_size
                    global_inodes.add(key)
        rows.append({'directory':directory.name, 'regular_files':files, 'logical_bytes':size,
                     'unique_inode_bytes':unique_size, 'symlinks':links,
                     'redistribution':'prohibited for BEHAVIOR assets/key' if directory.name=='behavior_1k'
                                      else 'subject to each upstream source license'})
    report = {'scope':'Canonical Assets directories; no following directory symlinks. '
                       'Unique-inode bytes are content length, not disk allocation or compression.',
              'unique_inode_bytes_total':global_bytes, 'directories':rows}
    output = WORLD/'reports/assets/inventory.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    text = ['# 本机资产清单', '', '本表记录文件就位情况，不等于场景通过；判据和视频见 reports/validation。共享依赖不是新增 benchmark。', '',
            '| 目录 | 文件数 | 按路径计 GiB | 去重内容 GiB |', '|---|---:|---:|---:|']
    for row in rows:
        text.append(f"| {row['directory']} | {row['regular_files']} | {row['logical_bytes']/2**30:.3f} | {row['unique_inode_bytes']/2**30:.3f} |")
    text += ['', '**BEHAVIOR 资产和密钥不得上传 Hugging Face；其他目录仍须按原始来源许可选择发布。**', '',
             '完整迁移与哈希记录在 reports/assets/；仍未取得的上游资产见逐任务核验报告。']
    (root/'INDEX.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({'directories':len(rows), 'unique_inode_bytes_total':global_bytes}))


if __name__ == '__main__':
    main()
