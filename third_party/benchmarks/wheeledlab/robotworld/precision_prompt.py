"""Public vehicle-course instructions; never export layout coordinates."""
def task(spec):
    if spec['case']=='rw-twin-beam':
        return ('Align both wheel tracks with the two narrow raised beams, cross the entire bridge and stop centred on the green landing mark. '
                'The beams are95mm wide and the tires about41mm wide, wheel track230mm. Solid yellow rail-edge paint is forbidden to tires; '
                'the vehicle body may span the gap between the beams. Missing a beam, touching rail-edge paint, falling or leaving the approach/landing fails. '
                'Use onboard views to align; no bridge coordinates or alignment error are supplied. Complete crossing then stop within0.14m of the green centre, '
                'aligned with the bridge within5deg, speed<=0.035m/s for1s. Reverse corrections are allowed.')
    prefix=('Park inside the yellow outlined EMPTY bay with the green centre mark. Solid yellow lines are strict forbidden boundaries: '
            'the whole declared0.60x0.36m safety footprint must not overlap them, or neighbouring parked vehicles. '
            'Only the WHITE DASHED mouth line is traversable. Keep inside the marked narrow manoeuvring aisle. '
            'You may move forward and reverse multiple times. You must actually cross into the bay while reversing, not enter forward and merely switch gear at rest. ')
    if spec['case']=='rw-reverse-bay':
        prefix+='This is perpendicular reverse-bay parking: finish nose pointing out of the bay toward the aisle. Reverse at least0.45m inside the bay. '
    else:
        prefix+='This is parallel parking between the front and rear parked cars: finish parallel to the neighbours, facing the same direction as at episode start. Reverse at least0.50m inside the bay. '
    return prefix+('Success requires the entire safety footprint inside the bay, centre within0.025m of the green mark, '
                   'heading error<=3deg, ground speed<=0.035m/s continuously for1s. Estimate position from front/rear onboard cameras; '
                   'no target coordinates, distance-to-line, completion progress or automatic parking controller is available.')
