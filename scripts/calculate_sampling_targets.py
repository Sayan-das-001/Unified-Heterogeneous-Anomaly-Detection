datasets = {
    "UNSW": {"normal":1959772, "attack":99643, "target":300000},
    "BoT": {"normal":5927447, "attack":513500, "target":200000},
    "CIC": {"normal":178379, "attack":7392035, "target":200000},
    "5G": {"normal":477736, "attack":738153, "target":300000}
}

for name,data in datasets.items():

    total = data["normal"] + data["attack"]

    normal_ratio = data["normal"]/total
    attack_ratio = data["attack"]/total

    normal_target = int(data["target"] * normal_ratio)
    attack_target = int(data["target"] * attack_ratio)

    print("\n",name)
    print("Target Samples:",data["target"])
    print("Normal:",normal_target)
    print("Attack:",attack_target)
