# Event Tracking

The `EventTracker` class monitors and records significant events that occur during robot manipulation tasks. It provides a unified mechanism to detect anomalies, failures, and state transitions during task execution.

## Overview

The tracker maintains internal state to detect transitions and ensures each event type is only recorded once per occurrence. When a condition clears (e.g., gripper lifts off table), the tracker resets for that event, allowing it to be recorded again if it reoccurs.

Events are returned as `(info_string, StatusCode)` tuples, where the info string provides human-readable context and the status code categorizes the event type.

## Tracked Event Modes

| Event | Description |
|-------|-------------|
| **WRONG_OBJECT_GRABBED** | Gripper grasped an object not in the intended target list |
| **GRIPPER_HIT_TABLE** | Gripper made contact with the table surface |
| **GRIPPER_FULLY_CLOSED** | Gripper closed completely without grasping an object (potential failed grasp) |
| **OBJECT_STARTED_MOVING** | A non-target object transitioned from stationary to moving |
| **OBJECT_BUMPED** | Object stopped after minor displacement (< move threshold) |
| **OBJECT_MOVED** | Object stopped after significant displacement (>= move threshold) |
| **OBJECT_OUT_OF_SCENE** | Object moved outside the defined workspace bounding box |
| **OBJECT_TIPPED_OVER** | An object that should remain upright has fallen over |
| **TARGET_OBJECT_DROPPED** | Target object was successfully grasped but released mid-transport |
| **GRIPPER_HIT_OBJECT** | Gripper collided with a non-target object |
| **MULTIPLE_OBJECTS_GRABBED** | A single gripper is in contact with multiple objects simultaneously |

### Multi-gripper robots

Contact events resolve gripper names through the robot's `contact_gripper` declaration
(see [Robots](robots.md#contact-gripper)), and the two contact events treat the
`"gripper"` alias group differently on purpose:

- **GRIPPER_HIT_OBJECT** uses the group as-is: touching a non-target object with *any*
  gripper is a hit, whichever hand did it.
- **MULTIPLE_OBJECTS_GRABBED** is evaluated **per concrete gripper**: the violation is one
  gripper clutching several objects at once. Contacts are counted separately for each
  member of the group, so a bimanual robot holding one object in each hand does not
  trigger it — but either hand individually touching two objects does.
