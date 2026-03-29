import json

reg = json.load(open('data/out/02_lineage_tracking/lineage_registry.json'))

# Find quarters where lineage 2 exists
quarters_with_lin2 = []
for quarter, community_map in reg.items():
    for comm_id, lin_id in community_map.items():
        if lin_id == 2:
            quarters_with_lin2.append((quarter, comm_id))
            break

print(f"Lineage 2 appears in {len(quarters_with_lin2)} quarters")
if quarters_with_lin2:
    q, cid = quarters_with_lin2[0]
    print(f"First occurrence: {q}, community {cid}")
    print(f"Types: community_id={type(cid)}, lineage_id={int}")
else:
    print("ERROR: Lineage 2 not found in any quarter!")
