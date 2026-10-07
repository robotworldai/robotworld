"""Create a labelled slow-playback copy; retain the original physics-rate video."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    report = json.loads((args.run/'physical-fixtures.json').read_text())
    source = args.run/'over_shoulder_left_camera-physics-fixtures.mp4'
    destination = args.run/'over_shoulder_left_camera-physics-fixtures-review-slow8x.mp4'
    fps = 1/report['physics_dt']
    filters = ['setpts=8*PTS',
               "drawtext=text='INJECTED STATES - 8x slower playback - NOT a policy rollout':x=20:y=20:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7"]
    start = 0
    for row in report['tests']:
        end = row['last_frame']
        qualifier = 'native success' if all(v==row['actual'] for v in row['native_values_by_physics_step']) else 'phase final native success'
        label = row['fixture']+' - '+qualifier+' '+str(row['actual']).upper()
        filters.append(f"drawtext=text='{label}':x=20:y=58:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.7:enable='gte(t,{start/fps*8})*lt(t,{end/fps*8})'")
        start = end
    command = ['ffmpeg','-v','error','-y','-i',str(source),'-vf',','.join(filters),
               '-r',str(fps/8),'-frames:v',str(report['frames']),'-c:v','libx264',
               '-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    subprocess.run(command, check=True)
    probe = subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries',
                            'stream=nb_frames,avg_frame_rate,duration','-of','json',str(destination)],
                           capture_output=True,text=True,check=True)
    stream = json.loads(probe.stdout)['streams'][0]
    if int(stream['nb_frames']) != report['frames']:raise RuntimeError('Slow review changed frame count')
    (args.run/'fixture-review.json').write_text(json.dumps({'source':str(source), 'review':str(destination),
        'playback_slowdown':8, 'original_simulated_seconds':report['frames']/fps,
        'same_frame_count':True, 'review_stream':stream, 'robot_policy':False},indent=2))
    print(destination)


if __name__ == '__main__':
    main()
