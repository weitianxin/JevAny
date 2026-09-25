# Thirty ways to act

The same JevAny-27B-SFT checkpoint chooses actions across the tasks below. The overview shows each selected action and its original probability.

[![Thirty JevAny tasks in one animated overview](demos/jevany-cases.gif)](demos/jevany-cases.gif)

## Robotics, assembly and laboratory automation

| Case | Task | Recorded outcome |
| --- | --- | --- |
| Robot assembly | Commission an inspection rover with a 24 V bus, a 5 V camera rail, inspection firmware, fresh wiring measurements and camera calibration. Mechanical access, power isolation and test validity constrain the order of work. | Passed |
| Laboratory automation | Calibrate the pipette, transfer 40 µL blue and 20 µL red into B3, and mix without contaminating the stocks. Calibration after a transfer does not repair a wrong dose; excessive mixing speed spills liquid. | Passed |
| Circuit-board inspection | Distinguish cosmetic surface marks from electrical discontinuity before diverting a board. Optical inspection and measured resistance must lead to the correct shipment, without discarding intact boards. | Passed |
| Drawer manipulation | Reach and grasp the handle, pull the drawer open by 14.86 cm, then release it. The close view exposes the sliding mechanism. | Passed |
| Peg insertion | Grasp the green peg, lift it over the cyan socket, lower it and release it to settle inside the opening. | Passed |
| Conveyor sorting | Advance and stop the belt, grasp the green parcel, and place it in the matching bin. All nine recorded choices are retained. | Passed |
| Contact manipulation | Choose a contact path around the barrier from seven persistent motion controls. Contact with the obstacle remains a failure even if the part later reaches the target. | Passed |
| Physical control panel | Use short and long physical button presses to isolate power, bleed pressure, clamp and commission a fixture at 20–30 kPa. Loaded clamping and overpressure create faults. Button travel is Bullet physics; pressure and interlocks use an explicit state model. | Failed attempt |
| Block stacking | Grasp and lift the green block, seat it on the blue base, and release the stack. The side view makes the contact height visible. | Passed |

## Mobility and fulfillment

| Case | Task | Recorded outcome |
| --- | --- | --- |
| Drone delivery | Plan for route energy, pad occupancy, waiting and landing, then deliver with at least 12 energy units in reserve. Waiting consumes energy; releasing before landing loses the payload. | Failed attempt |
| Warehouse fulfillment | Navigate to the rack, load the matching SKU, scan its barcode and secure the fragile parcel before padded dispatch. Wrong cargo and transit damage persist into the receipt. | Passed |
| Roadworks bypass | Choose among eight persistent controls using measured geometry. Drive around the roadworks, return and stop without obstacle contact. | Passed |
| Warehouse navigation | Choose among eight waypoints from the current pose and shelf geometry. The Husky must reach its dock without contact; candidate names do not reveal the safe sequence. | Passed |
| Pedestrian yielding | The pedestrian moves independently as simulated time advances. Choose where to wait and for how long; entering while occupied remains a violation after the pedestrian clears. | Passed |
| Fleet dispatch | Coordinate payload, cold-chain temperature, range budget and bridge clearance. Repeated dispatches consume resources, so a correct vehicle assignment alone does not complete both deliveries. | Failed attempt |
| City-road driving | Treat crosswalk clearance and the rear-lane gap as separate observations. Use a current gap measurement to merge safely, reach the north exit and stop. | Failed attempt |
| Reverse parking | Choose incremental reverse or forward motions using measured bay and vehicle bounds. The entire vehicle must fit between the parked cars, rather than only its reference point. | Passed |

## Geospatial analysis, services and tactical decisions

| Case | Task | Recorded outcome |
| --- | --- | --- |
| Geospatial change analysis | Register the offset images, mask clouds and measure vegetation change before classification. Export the correct region and coordinate reference as actual GeoJSON. | Passed |
| Selective threat response | An abusive source IP also carries five legitimate sessions. Compare session policies and replay traffic before deployment, preserving both browser and non-browser legitimate clients. | Passed |
| Service recovery | One current replica lacks full-load capacity; another has stale data. Prepare a replica, test a canary, promote traffic, probe the full load and drain the failed primary without losing the rollout error budget. | Passed |
| Tactical chess | Find a forced mate in two in a nine-piece position. All legal moves are available without checkmate labels; a defending search chooses a legal reply that avoids mate when possible. | Failed attempt |

## Browser work and productivity

| Case | Task | Recorded outcome |
| --- | --- | --- |
| Procurement browsing | Compare hardware, commercial licensing, stock, shipping cost and a three-day deadline against a 650 total budget. Changing the product invalidates the prior quote assumptions. | Passed |
| Calendar coordination | Convert attendee time zones and include 15-minute buffers. Coordinate room occupancy, capacity and a portable display kit, refreshing availability after each draft change. | Failed attempt |
| Web research | Resolve conflicting documentation versions, cite both the current reference and release notes, and run the local command-argument checker before saving the evidence. | Passed |
| Inbox triage | Read the latest reply and verify the sender before filing. An urgent subject may already be resolved; a receipt subject may conceal a failed payment and deadline. | Failed attempt |
| Spreadsheet cleanup | Keep the latest order revision, parse each row using its own locale and reconcile missing quantities with a shipping ledger. Validate the revenue and export the real cleaned CSV. | Passed |

## Software and data workflows

| Case | Task | Recorded outcome |
| --- | --- | --- |
| Frontend repair | Repair card width, shrinking child elements and touch-target size separately. Chromium measures visibility, overflow and 44 px targets at 320, 390 and 680 px; stale measurements cannot authorize publication. | Passed |
| SQL repair | Repair aggregation on actual SQLite data containing equal order amounts, NULLs, drafts and one-to-many lines. Both EXISTS and pre-aggregation can work; SUM(DISTINCT) loses legitimate equal-value orders. | Passed |
| Build and release recovery | Package the entry point, settings and selected locale assets, then execute the bundle smoke test. A configuration change after testing requires a fresh build and test before staging. | Passed |
| Database performance | Preserve the fresh, complete SQLite result with at most six queries and 24 rows per fetch. A single large batch violates the memory bound; stale caches and limited results fail correctness. Chunking and streaming are both supported. | Passed |

## Recording notes

The [decision record](demos/case-showcase.json) contains all 57 current attempts and 642 decisions, including the 248 choices shown in the overview. It preserves each candidate distribution, supplied text state, selected action and resulting state. Original observation images and complete physics trajectories remain in the local recording library.

Twenty-six tasks were revised and rerun; 19 were accepted within the predefined trials. Four existing contact tasks retain their original runs. The first successful attempt in each predefined seed list is shown. If all three attempts fail, the first failed attempt is shown and labeled.

The 21 labeled 3D replays reconstruct recorded state and motion; they are not the model’s original observations. The nine interface cases use actual browser recordings. Revised Bullet tasks were rerun with physical checks; added presentation geometry does not participate in those checks. These controlled examples illustrate integrations and do not constitute a general capability benchmark.

[Back to the README](../README.md) · [Benchmark evaluations](../README.md#results)
