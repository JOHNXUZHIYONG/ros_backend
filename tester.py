figure_names = []
input_data = { "data": { "id": 2, "use_case": "indoor", \
        "thrVsCommRadius": False, "effVsCommRadius": True, \
            "effVsTxPower": True, "latVsCommRadius": False, \
                "latVsTaskSize": False, "task_size_value": 10, \
                    "bandwidth_value": 10, "speed_value": 20, \
                        "hops_value": 5, "comm_rad_value": 100, "num_sam_value": 50, "email": "john@email" } }
data = input_data.get("data")

for index, (key,value) in enumerate(data.items()):
    
    # get figure name if user wants figure to be plotted
    figure_names.append(key) if value == True else print("")

figure_names = [figures for figures in list(data.get("data")) if figures.value()==True]


# figure_names: List[str],
# email_addresses: List[str],
# plot_parameters: Dict[str, List[float]]