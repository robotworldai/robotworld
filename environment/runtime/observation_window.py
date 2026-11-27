"""Trim only older RobotWorld observation images on the outbound Responses wire.
The original native conversation and archived tool results remain untouched.
"""
import copy
import json

NOTICE = '[Earlier RobotWorld observation image omitted; use the newest observation packet. Full image retained in the trajectory archive.]'

def image_count(value):
    if isinstance(value, list):return sum(map(image_count,value))
    if isinstance(value, dict):
        if value.get('type')=='input_image':return 1
        return sum(image_count(v) for v in value.values())
    return 0


def trim_observations(body):
    if body.get('previous_response_id') or body.get('conversation'):
        raise ValueError('Image-window policy requires explicit stateless Responses history')
    out=copy.deepcopy(body);items=out.get('input',[])
    packets=[]
    auxiliary=[]
    calls={item.get('call_id'):item.get('name') for item in items
           if isinstance(item,dict) and item.get('type')=='function_call'} if isinstance(items,list) else {}
    if isinstance(items,list):
        for index,item in enumerate(items):
            if item.get('type')!='function_call_output' or not isinstance(item.get('output'),list):continue
            parts=item['output'];metadata=[]
            for part in parts:
                if part.get('type') not in ('input_text','text'):continue
                try:data=json.loads(part.get('text',''))
                except (ValueError,TypeError):continue
                if (isinstance(data,dict) and data.get('protocol')=='robotworld.episode.v1'
                    and isinstance(data.get('episode_id'),str)
                    and type(data.get('control_step')) is int
                    and isinstance(data.get('observation'),dict)):
                    metadata.append(data)
            if image_count(parts):
                if not metadata and calls.get(item.get('call_id')) in ('view_image', 'functions.view_image'):
                    auxiliary.extend((index, i) for i, part in enumerate(parts) if part.get('type')=='input_image')
                    continue
                if len(metadata)!=1:raise ValueError('Unmarked image-bearing tool output; cannot guarantee observation window')
                packets.append((index,metadata[0]['episode_id'],metadata[0]['control_step']))
    keep=None
    if packets:
        if len({x[1] for x in packets})!=1:raise ValueError('Mixed episode images in one request')
        # Physical time wins; latest duplicate at the same step wins.
        keep=max(packets,key=lambda x:(x[2],x[0]))
        for index,_,_ in packets:
            if index==keep[0]:continue
            parts=items[index]['output']
            items[index]['output']=[{'type':'input_text','text':NOTICE} if p.get('type')=='input_image' else p for p in parts]
    # Native view_image returns image-only output, not a simulator observation.
    # Keep a bounded auxiliary window without changing archived tool results.
    for index, part_index in auxiliary[:-4]:
        items[index]['output'][part_index]={'type':'input_text','text':'[Earlier auxiliary image omitted; full image retained in the trajectory archive.]'}
    after=image_count(out)
    if after>50:raise ValueError('Newest packet plus other images exceeds the 50-image safety cap')
    return out,{'images_before':image_count(body),'images_after':after,
                'auxiliary_images_before':len(auxiliary),'auxiliary_images_after':min(4,len(auxiliary)),
                'packets_before':len(packets),'kept_control_step':keep[2] if keep else None,
                'kept_call_id':items[keep[0]].get('call_id') if keep else None}
