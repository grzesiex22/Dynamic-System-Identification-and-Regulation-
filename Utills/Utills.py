def get_unique_signal_indices(dataset_name, mode="test", noise_level=0.0):
    """
    Przeszukuje raport JSON i zwraca pierwsze wystąpienie każdego typu sygnału.
    Zawsze szuka bazowego pliku info.json (bez szumu), bo struktura sygnałów jest ta sama.
    """
    import json
    import glob
    import os

    # Szukamy podstawowego pliku info.json dla danego trybu (np. test_dataset_*_info.json)
    # Ignorujemy poziom szumu w nazwie pliku JSON, bo struktura trajektorii jest identyczna
    pattern = os.path.join("Dataset", dataset_name, f"{mode}_dataset_*_info.json")
    all_info_files = glob.glob(pattern)

    # Filtrujemy, aby wziąć plik główny (ten bez "_noise_" w nazwie), 
    # ponieważ on zawiera pierwotną historię generowania sygnałów
    base_info_files = [f for f in all_info_files if "_noise_" not in os.path.basename(f)]

    if not base_info_files:
        # Jeśli nie ma pliku bez "noise", bierzemy jakikolwiek info.json który jest pod ręką
        if all_info_files:
            json_path = all_info_files[0]
        else:
            print(f"⚠️ OSTRZEŻENIE: Nie znaleziono pliku info.json w {dataset_name}")
            return {}
    else:
        json_path = base_info_files[0]

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        history = report.get("run_history", [])
        found_types = {}
        for idx, entry in enumerate(history):
            s_type = entry["type"]
            if s_type not in found_types:
                found_types[s_type] = idx
            if len(found_types) == 3: break

        return found_types
    except Exception as e:
        print(f"❌ Błąd podczas odczytu JSON: {e}")
        return {}