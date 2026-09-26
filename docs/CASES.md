# Cases

JevAny-27B-SFT chooses the actions in every case below. The overview shows each selected action with the probability it carried in the original run.

[![JevAny tasks in one animated overview](demos/jevany-cases.gif)](demos/jevany-cases.gif)

## Robotics, assembly and laboratory automation

| Case | Task | Successful run |
| --- | --- | --- |
| Robot assembly | Commission an inspection rover with a 24 V bus, a 5 V camera rail, inspection firmware, fresh wiring measurements and camera calibration. Mechanical access, power isolation and test validity constrain the order of work. | [![Robot assembly — successful run](demos/cases/assembly.gif)](demos/cases/assembly.gif) |
| Laboratory automation | Calibrate the pipette, transfer 40 µL blue and 20 µL red into B3, and mix without contaminating the stocks. Calibration after a transfer does not repair a wrong dose; excessive mixing speed spills liquid. | [![Laboratory automation — successful run](demos/cases/lab.gif)](demos/cases/lab.gif) |
| Circuit-board inspection | Distinguish cosmetic surface marks from electrical discontinuity before diverting a board. Optical inspection and measured resistance must lead to the correct shipment, without discarding intact boards. | [![Circuit-board inspection — successful run](demos/cases/quality.gif)](demos/cases/quality.gif) |
| Drawer manipulation | Reach and grasp the handle, pull the drawer open by 14.86 cm, then release it. The close view exposes the sliding mechanism. | [![Drawer manipulation — successful run](demos/cases/drawer_opening.gif)](demos/cases/drawer_opening.gif) |
| Peg insertion | Grasp the green peg, lift it over the cyan socket, lower it and release it to settle inside the opening. | [![Peg insertion — successful run](demos/cases/peg_insertion.gif)](demos/cases/peg_insertion.gif) |
| Conveyor sorting | Advance and stop the belt, grasp the green parcel, and place it in the matching bin. | [![Conveyor sorting — successful run](demos/cases/conveyor_sorting.gif)](demos/cases/conveyor_sorting.gif) |
| Contact manipulation | Choose a contact path around the barrier from seven persistent motion controls. Touching the obstacle counts against the run even if the part later reaches the target. | [![Contact manipulation — successful run](demos/cases/obstacle_push.gif)](demos/cases/obstacle_push.gif) |
| Physical control panel | Use short and long physical button presses to isolate power, bleed pressure, clamp and commission a fixture at 20–30 kPa. Loaded clamping and overpressure create faults. Button travel is Bullet physics; pressure and interlocks use an explicit state model. | [![Physical control panel — successful run](demos/cases/control_panel.gif)](demos/cases/control_panel.gif) |
| Block stacking | Grasp and lift the green block, seat it on the blue base, and release the stack. The side view makes the contact height visible. | [![Block stacking — successful run](demos/cases/block_stacking.gif)](demos/cases/block_stacking.gif) |

## Mobility and fulfillment

| Case | Task | Successful run |
| --- | --- | --- |
| Drone delivery | Plan for route energy, pad occupancy, waiting and landing, then deliver with at least 12 energy units in reserve. Waiting consumes energy; releasing before landing loses the payload. | [![Drone delivery — successful run](demos/cases/drone.gif)](demos/cases/drone.gif) |
| Warehouse fulfillment | Navigate to the rack, load the matching SKU, scan its barcode and secure the fragile parcel before padded dispatch. Wrong cargo and transit damage persist into the receipt. | [![Warehouse fulfillment — successful run](demos/cases/warehouse.gif)](demos/cases/warehouse.gif) |
| Roadworks bypass | Choose among eight persistent controls using measured geometry. Drive around the roadworks, return and stop without obstacle contact. | [![Roadworks bypass — successful run](demos/cases/lane_change.gif)](demos/cases/lane_change.gif) |
| Warehouse navigation | Choose among eight waypoints from the current pose and shelf geometry. The Husky must reach its dock without contact; candidate names do not reveal the safe sequence. | [![Warehouse navigation — successful run](demos/cases/warehouse_rover.gif)](demos/cases/warehouse_rover.gif) |
| Pedestrian yielding | The pedestrian moves independently as simulated time advances. Choose where to wait and for how long; entering while occupied remains a violation after the pedestrian clears. | [![Pedestrian yielding — successful run](demos/cases/crosswalk.gif)](demos/cases/crosswalk.gif) |
| Fleet dispatch | Coordinate payload, cold-chain temperature, range budget and bridge clearance. Repeated dispatches consume resources, so a correct vehicle assignment alone does not complete both deliveries. | [![Fleet dispatch — successful run](demos/cases/fleet.gif)](demos/cases/fleet.gif) |
| City-road driving | Treat crosswalk clearance and the rear-lane gap as separate observations. Use a current gap measurement to merge safely, reach the north exit and stop. | [![City-road driving — successful run](demos/cases/driving.gif)](demos/cases/driving.gif) |
| Reverse parking | Choose incremental reverse or forward motions using measured bay and vehicle bounds. The entire vehicle must fit between the parked cars, rather than only its reference point. | [![Reverse parking — successful run](demos/cases/reverse_parking.gif)](demos/cases/reverse_parking.gif) |

## Geospatial analysis, services and tactical decisions

| Case | Task | Successful run |
| --- | --- | --- |
| Geospatial change analysis | Register the offset images, mask clouds and measure vegetation change before classification. Export the correct region and coordinate reference as actual GeoJSON. | [![Geospatial change analysis — successful run](demos/cases/satellite.gif)](demos/cases/satellite.gif) |
| Selective threat response | An abusive source IP also carries five legitimate sessions. Compare session policies and replay traffic before deployment, preserving both browser and non-browser legitimate clients. | [![Selective threat response — successful run](demos/cases/security.gif)](demos/cases/security.gif) |
| Service recovery | One current replica lacks full-load capacity; another has stale data. Prepare a replica, test a canary, promote traffic, probe the full load and drain the failed primary without losing the rollout error budget. | [![Service recovery — successful run](demos/cases/service.gif)](demos/cases/service.gif) |
| Tactical chess | Find a forced mate in two in a nine-piece position. All legal moves are available without checkmate labels; a defending search chooses a legal reply that avoids mate when possible. | [![Tactical chess — successful run](demos/cases/chess.gif)](demos/cases/chess.gif) |

## Browser work and productivity

| Case | Task | Successful run |
| --- | --- | --- |
| Procurement browsing | Compare hardware, commercial licensing, stock, shipping cost and a three-day deadline against a 650 total budget. Changing the product invalidates the prior quote assumptions. | [![Procurement browsing — successful run](demos/cases/procurement.gif)](demos/cases/procurement.gif) |
| Calendar coordination | Convert attendee time zones and include 15-minute buffers. Coordinate room occupancy, capacity and a portable display kit, refreshing availability after each draft change. | [![Calendar coordination — successful run](demos/cases/calendar.gif)](demos/cases/calendar.gif) |
| Web research | Resolve conflicting documentation versions, cite both the current reference and release notes, and run the local command-argument checker before saving the evidence. | [![Web research — successful run](demos/cases/research.gif)](demos/cases/research.gif) |
| Inbox triage | Read the latest reply and verify the sender before filing. An urgent subject may already be resolved; a receipt subject may conceal a failed payment and deadline. | [![Inbox triage — successful run](demos/cases/inbox.gif)](demos/cases/inbox.gif) |
| Spreadsheet cleanup | Keep the latest order revision, parse each row using its own locale and reconcile missing quantities with a shipping ledger. Validate the revenue and export the real cleaned CSV. | [![Spreadsheet cleanup — successful run](demos/cases/spreadsheet.gif)](demos/cases/spreadsheet.gif) |

## Software and data workflows

| Case | Task | Successful run |
| --- | --- | --- |
| Frontend repair | Repair card width, shrinking child elements and touch-target size separately. Chromium measures visibility, overflow and 44 px targets at 320, 390 and 680 px; stale measurements cannot authorize publication. | [![Frontend repair — successful run](demos/cases/frontend.gif)](demos/cases/frontend.gif) |
| SQL repair | Repair aggregation on actual SQLite data containing equal order amounts, NULLs, drafts and one-to-many lines. Both EXISTS and pre-aggregation can work; SUM(DISTINCT) loses legitimate equal-value orders. | [![SQL repair — successful run](demos/cases/sql.gif)](demos/cases/sql.gif) |
| Build and release recovery | Package the entry point, settings and selected locale assets, then execute the bundle smoke test. A configuration change after testing requires a fresh build and test before staging. | [![Build and release recovery — successful run](demos/cases/ci.gif)](demos/cases/ci.gif) |
| Database performance | Preserve the fresh, complete SQLite result with at most six queries and 24 rows per fetch. A single large batch violates the memory bound; stale caches and limited results fail correctness. Chunking and streaming are both supported. | [![Database performance — successful run](demos/cases/performance.gif)](demos/cases/performance.gif) |

## Recording notes

These are selected successful examples from repeated attempts; failed attempts remain in the records. The left panel shows the original candidate probabilities and recorded action replay progress, with execution feedback after each step. Reruns correct stale interface observations and clarify action preconditions while keeping task mechanics and acceptance checks unchanged. Sampled examples (Physical control panel, Tactical chess) use the checkpoint’s unmodified action distribution; their GIFs label this policy, and the record also retains the highest-probability action.

The [decision record](demos/case-showcase.json) holds every attempt behind these cases: each candidate distribution, the supplied text state, the selected action and the resulting state. Original observation images and complete physics trajectories stay in the local recording library.

The tiles labeled as 3D replays reconstruct recorded state and motion, so they are not the model's own observations, and geometry added for presentation takes no part in the physical checks. The interface cases use real browser recordings. These are controlled examples of the integrations, so read them as illustrations; measured accuracy is in [the benchmark evaluations](EVALUATION.md).

[Back to the README](../README.md)
