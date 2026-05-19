import copy
import sys
from colorama import Fore, Back, Style, init
import numpy as np
from matplotlib import pyplot as plt

from Test.Tester import Tester
from Utills.SystemPlotter import SystemPlotter
from Utills.Utills import get_unique_signal_indices
from Dataset.DatasetCreator import DatasetCreator
from Dataset.DatasetReader import DatasetReader

from ML.TorchSytsemMLP import TorchSytsemMLP
from Objects.CoupledTanks import CoupledTanks1
from Test.Metrics import Metrics, MetricsSummarizer
from ML.ImplementationOfMLP import OwnSystemMLP
from ML.SklearnSystemMLP import SklearnSystemMLP
from ML.KerasSystemMLP import KerasSystemMLP
from ML.TorchLSTMSystem import TorchLSTMSystem
from ML.RegressionSystemModel import RegressionSystemModel

# Inicjalizacja colorama (autoreset sprawia, że kolor wraca do normy po każdym princie)
init(autoreset=True)


# -----------------------------
# 0. Parametry
# -----------------------------
t_end = 1000
dt = 0.5
amp_range = (3, 15)

# --- generator
GENERATE = False
train_dataset_count = 1000
val_dataset_count = 200
test_dataset_count = 200
noise_level = 0.05
dataset_name = "Dataset_test_1"

# --- model - ogólne zmienne ---
SHOWCASE = False
TRAIN_AND_SAVE = False
LOAD = True
TEST = True
FINAL_PLOTS = True
epochs = 500
patience = 20

# --- Konfiguracja modeli ---
# Słownik konfiguracji model
models_config = [
    {"obj": TorchSytsemMLP(input_dim=5, hidden_dim=128, output_dim=2), "name": "Torch_MLP"},
    {"obj": OwnSystemMLP(input_dim=5, hidden_dim=128, output_dim=2), "name": "Own_MLP"},
    # {"obj": SklearnSystemMLP(input_dim=5, hidden_dim=128, output_dim=2), "name": "Sklearn_MLP"},
    # {"obj": KerasSystemMLP(input_dim=5, hidden_dim=128, output_dim=2), "name": "Keras_MLP"},
    {"obj": TorchLSTMSystem(input_dim=5, hidden_dim=64, output_dim=2, seq_len=10, num_layers=1), "name": "Torch_LSTM"},
    {"obj": RegressionSystemModel(input_dim=5, output_dim=2, alpha=1.0), "name": "Ridge_Regression"},
]

models = []

# --- wykresy ---
show_showcase_plot = False
show_learning_plot = False
show_testing_plot = True


# --------------------------------------------------------------------------------------------------------------------
# 1. Definicja obiektu dynamicznego
# --------------------------------------------------------------------------------------------------------------------
print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*30} INICJALIZACJA SYSTEMU {'='*30}")
tanks = CoupledTanks1(t_end=t_end)
print(f"{Fore.GREEN}✅ Obiekt {Fore.WHITE}{Style.BRIGHT}{tanks.__class__.__name__}{Fore.GREEN} został pomyślnie zainicjalizowany.")


# --------------------------------------------------------------------------------------------------------------------
# 2. Generator przebiegów
# --------------------------------------------------------------------------------------------------------------------
print(f"\n{Fore.MAGENTA}{Style.BRIGHT}{'─'*100}")
print(f"{Fore.MAGENTA}{Style.BRIGHT}🛠️  MODUŁ GENERACJI DANYCH")
print(f"{Fore.MAGENTA}{Style.BRIGHT}{'─'*100}")

if GENERATE:
    dataset_creator = DatasetCreator(tanks, t_end=t_end, dt=dt, amp_range=amp_range, noise_level=noise_level)

    # Lista konfiguracji do wygenerowania (ułatwia iterację i ładny print)
    tasks = [
        {"count": train_dataset_count, "mode": "train", "label": "TRENINGOWEGO"},
        {"count": val_dataset_count, "mode": "val", "label": "WALIDACYJNEGO"},
        {"count": test_dataset_count, "mode": "test", "label": "TESTOWEGO"}
    ]

    for task in tasks:
        print(
            f"\n{Fore.YELLOW}⏳ Generowanie zbioru {Fore.WHITE}{Style.BRIGHT}{task['label']} {Fore.YELLOW}(n={task['count']})...")

        status = dataset_creator.create_dataset(
            n_trajectories=task['count'],
            mode=task['mode'],
            folder=dataset_name
        )

        if status != 0:
            print(
                f"\n{Fore.RED}{Back.WHITE}{Style.BRIGHT} ❌ KRYTYCZNY BŁĄD: Generowanie przerwane na etapie: {task['mode']} ")
            print(f"{Fore.RED}Podpowiedź: Sprawdź czy folder '{dataset_name}' nie zawiera już tych plików.")
            sys.exit(1)

    print(
        f"\n{Fore.GREEN}{Style.BRIGHT}✨ PROCES ZAKOŃCZONY: Wszystkie zestawy danych są gotowe w folderze '{dataset_name}'!")

    print(f"\n{Fore.CYAN}📺 Przygotowuję podgląd sygnałów sterujących...")
    dataset_creator.show_random_signals(3)

else:
    print(f"{Fore.BLUE}ℹ️  Pomijanie generowania danych (flaga 'generate' jest ustawiona na False).")
    print(f"{Fore.BLUE}📂 Program przejdzie bezpośrednio do odczytu z folderu: {Fore.WHITE}{Style.BRIGHT}{dataset_name}")

print(f"{Fore.MAGENTA}{Style.BRIGHT}{'─' * 100}\n")

# --------------------------------------------------------------------------------------------------------------------
# 3. Weryfikacja datasetu
# --------------------------------------------------------------------------------------------------------------------

print(f"\n{Fore.CYAN}{Style.BRIGHT}{'═' * 30} Weryfikacja datasetu do odczytu {'═' * 30}")

# --- Sprawdzenie DATASET ---
reader = DatasetReader()
available_variants = reader.check_available_variants(dataset_name)

if not available_variants:
    print(Fore.RED + Style.BRIGHT + f"❌ BŁĄD: W folderze '{dataset_name}' brak danych!")
    sys.exit(1)

print(Fore.CYAN + f"🔎 Wykryte warianty: {Style.BRIGHT}{[v['name'] for v in available_variants]}")

# --------------------------------------------------------------------------------------------------------------------
# 4. Wczytanie wszystkich datasetów do pamięci
# --------------------------------------------------------------------------------------------------------------------
print(f"\n{Fore.CYAN}{Style.BRIGHT}{'═' * 30} 4. ŁADOWANIE DANYCH {'═' * 34}")

# Słownik do przechowywania wszystkich danych w pamięci RAM
loaded_data = {}

for var in available_variants:
    v_name = var["name"]
    v_noise = var["noise"]

    print(Fore.YELLOW + f"📂 Ładowanie wariantu: {Style.BRIGHT}{v_name}...", end=" ", flush=True)

    # Odczyt obiektów
    train_obs = reader.find_and_read(dataset_name, "train", noise_level=v_noise)
    val_obs = reader.find_and_read(dataset_name, "val", noise_level=v_noise)
    test_obs = reader.find_and_read(dataset_name, "test", noise_level=v_noise)

    # Konwersja na macierze numpy gotowe do treningu
    X_train = np.stack([obj.get_training_data()[0] for obj in train_obs])
    Y_train = np.stack([obj.get_training_data()[1] for obj in train_obs])
    X_val = np.stack([obj.get_training_data()[0] for obj in val_obs])
    Y_val = np.stack([obj.get_training_data()[1] for obj in val_obs])
    X_test = np.stack([obj.get_training_data()[0] for obj in test_obs])
    Y_test = np.stack([obj.get_training_data()[1] for obj in test_obs])

    # Zapis do słownika
    loaded_data[v_name] = {
        "noise": v_noise,
        "train_obs": train_obs,
        "val_obs": val_obs,
        "test_obs": test_obs,
        "X_train": X_train,
        "Y_train": Y_train,
        "X_val": X_val,
        "Y_val": Y_val,
        "X_test": X_test,
        "Y_test": Y_test
    }

    print(Fore.GREEN + Style.BRIGHT + "ZAŁADOWANO ✅")

    # Czytelne printy korzystające z lokalnych zmiennych
    print(Fore.MAGENTA + f"\nX_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}")
    print(Fore.MAGENTA + f"X_val shape: {X_val.shape}, Y_val shape: {Y_val.shape}")
    print(Fore.MAGENTA + f"X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}")


# --------------------------------------------------------------------------------------------------------------------
# 5. Showcase (Podgląd typów sygnałów: Clean vs Noisy)
# --------------------------------------------------------------------------------------------------------------------
print(f"\n{Fore.CYAN}{Style.BRIGHT}{'═' * 30} 5. SHOWCASE DANYCH (INTEGRITY CHECK) {'═' * 15}")

if SHOWCASE:
    try:
        # Szukamy, czy w ogóle mamy wariant z szumem do porównania
        noise_variants = [v for v in available_variants if v["noise"] > 0]

        # Zawsze bierzemy bazę CLEAN
        clean_name = "CLEAN"

        if clean_name in loaded_data and noise_variants:
            v_noise = noise_variants[-1]["noise"]  # Bierzemy największy szum do pokazu
            noisy_name = f"NOISY_{v_noise}"  # Dostosuj, jeśli Twoje nazwy wariantów są inne

            # Upewniamy się, że szukany wariant jest w słowniku
            if noisy_name in loaded_data:
                print(
                    f"{Fore.YELLOW}🔍 Porównanie typów sygnałów: {Fore.BLUE}CLEAN {Fore.YELLOW}vs {Fore.RED}{noisy_name}")

                # Wykorzystujemy Twoją funkcję do znalezienia indeksów (aprbs, multisine, noise)
                target_indices = get_unique_signal_indices(dataset_name, mode="test", noise_level=0.0)

                clean_test_obs = loaded_data[clean_name]["test_obs"]
                noisy_test_obs = loaded_data[noisy_name]["test_obs"]

                for sig_type, idx in target_indices.items():
                    print(f"  {Fore.GREEN}└─ Generowanie podglądu dla: {Style.BRIGHT}{sig_type.upper()} (index: {idx})")

                    c_obj = clean_test_obs[idx]
                    n_obj = noisy_test_obs[idx]

                    t_plot, u_plot, h_clean, dh_dt_clean = c_obj.get_data_to_plot()
                    _, _, h_noisy, dh_dt_noisy = n_obj.get_data_to_plot()

                    v_noise_str = str(noisy_name).replace('.', '_')  # Zamiana 0.5 na 0_5 (bezpieczniej w nazwach plików)

                    SystemPlotter.plot_noise_comparison(
                        t=t_plot,
                        u=u_plot,
                        y_true=h_clean,
                        dy_dt_true=dh_dt_clean,
                        y_noise=h_noisy,  # Bezpośrednio macierz
                        dy_dt_noise=dh_dt_noisy,  # Bezpośrednio macierz
                        noise_label=f"Zaszumione (std={v_noise})",
                        title=f"Showcase | Typ: {sig_type.upper()} | Clean vs {noisy_name}",
                        save_name=f"Showcase_t{idx}_{v_noise_str}_U_{sig_type.upper()}",
                        dataset=dataset_name,
                        show=show_showcase_plot
                    )
                if show_showcase_plot:
                    print(f"{Fore.CYAN}📺 Zamknij wykresy, aby rozpocząć proces uczenia/testowania.")
                    plt.show(block=True)
            else:
                print(f"{Fore.BLUE}ℹ️  Wariant {noisy_name} nie został załadowany. Pomijam showcase.")
        else:
            print(f"{Fore.BLUE}ℹ️  Brak danych CLEAN lub NOISY do wykonania porównania.")

    except Exception as e:
        print(f"{Fore.RED}⚠️ Błąd podczas weryfikacji Showcase: {Fore.WHITE}{e}")
        import traceback

        traceback.print_exc()
else:
    print(f"{Fore.YELLOW}Flaga SHOWCASE = {SHOWCASE} - brak pokazu przykładowych wczytanych sygnałów{Style.RESET_ALL}")

# --------------------------------------------------------------------------------------------------------------------
# 6. Pętla Główna (Trening/Wczytanie)
# --------------------------------------------------------------------------------------------------------------------
print(f"\n{Fore.CYAN}{Style.BRIGHT}{'═' * 30} 6. TRENING/WCZYTANIE MODELI {'═' * 19}")

# --- Weryfikacja flag logicznych ---
if TRAIN_AND_SAVE and LOAD:
    print(f"\n{Fore.RED}{Style.BRIGHT}❌ KRYTYCZNY BŁĄD KONFIGURACJI:")
    print(f"{Fore.RED}Zmienne 'TRAIN_AND_SAVE' oraz 'LOAD' nie mogą być obie ustawione na True!")
    print(f"{Fore.YELLOW}Zdecyduj czy chcesz trenować nowe modele, czy wczytać już istniejące.{Style.RESET_ALL}")
    sys.exit(1)

if not TRAIN_AND_SAVE and not LOAD:
    print(f"\n{Fore.BLUE}{Style.BRIGHT}ℹ️  INFORMACJA:")
    print(f"{Fore.BLUE}Obie flagi ('TRAIN_AND_SAVE', 'LOAD') są ustawione na False.")
    print(f"{Fore.BLUE}Program nie ma zadań do wykonania (brak treningu i brak wczytywania). Koniec.{Style.RESET_ALL}")
    sys.exit(0)

# Iterujemy po wczytanych wcześniej danych
for v_name, data in loaded_data.items():

    print("\n" + Fore.BLUE + Style.BRIGHT + "═" * 80)
    print(Fore.BLUE + Style.BRIGHT + f" 🚀 WARIANT: {v_name.center(66)} 🚀")
    print(Fore.BLUE + Style.BRIGHT + "═" * 80)

    # Wymiary z pamięci (tylko do logów)
    print(
        Fore.MAGENTA + f"📊 Dane X: Train={data['X_train'].shape}, Val={data['X_val'].shape}, Test={data['X_test'].shape}")
    print(
        Fore.MAGENTA + f"📊 Dane Y: Train={data['Y_train'].shape}, Val={data['Y_val'].shape}, Test={data['Y_test'].shape}")

    # --- Uczenie / wczytanie modeli ---
    for m_cfg in models_config:
        # Tworzymy nową, niezależną instancję modelu dla tego konkretnego wariantu danych
        model_obj = copy.deepcopy(m_cfg["obj"])
        model_name = m_cfg["name"]

        print(f"\n{Fore.WHITE + Back.BLUE + Style.BRIGHT} 🧠 MODEL: {model_name} ")
        save_id = f"{dataset_name}_{v_name}_{model_name}"

        if LOAD:
            print(Fore.CYAN + "  📥 Wczytywanie wag...", end=" ", flush=True)
            model_obj.load_model(dataset=dataset_name, base_name=save_id)
            print(Fore.GREEN + "Sukces")

        if TRAIN_AND_SAVE:
            print(Fore.YELLOW + "  🏗️  Rozpoczynam proces treningu...")
            model_obj.train(data["X_train"], data["Y_train"], data["X_val"], data["Y_val"],
                            epochs=epochs, patience=patience)
            model_obj.save_model(dataset=dataset_name, base_name=save_id)
            print(Fore.GREEN + f"  💾 Zapisano model: {save_id}")

            # --- Rysowanie krzywej uczenia ---
            loss_save_name = f"Loss_{model_name}_{v_name}"
            loss_save_name = loss_save_name.replace('.', '_')  # Zamiana 0.5 na 0_5 (bezpieczniej w nazwach plików)

            SystemPlotter.plot_learning_curves(
                history=model_obj.training_history,
                model_name=model_name,
                dataset=dataset_name,
                v_name=v_name,
                save_name=loss_save_name,
                show=show_learning_plot
            )

        # Dodajemy gotowy model do listy 'models'
        # Zapisujemy go z unikalną nazwą, żeby wiedzieć który jest który (np. "Torch_LSTM_NOISE_0.05")
        models.append({
            "obj": model_obj,
            "name": f"{model_name}_{v_name}",
            "variant": v_name,
            "base_name": model_name
        })

print(f"\n{Fore.GREEN}✅ Proces zakończony. Liczba aktywnych modeli w pamięci: {len(models)}")

# --------------------------------------------------------------------------------------------------------------------
# 7. Pętla Główna (Testowanie)
# --------------------------------------------------------------------------------------------------------------------
print(f"\n{Fore.CYAN}{Style.BRIGHT}{'═' * 25} 7. TESTOWANIE MODELI {'═' * 25}")

global_summarizer_dhdt = MetricsSummarizer()
global_summarizer_h = MetricsSummarizer()

if TEST:
    # Pobieramy referencyjne obiekty testowe (te bez szumu)
    clean_test_obs = loaded_data.get("CLEAN", {}).get("test_obs", None)

    # Iterujemy po wczytanych wcześniej danych
    for v_name, data in loaded_data.items():
        print("\n" + Fore.BLUE + Style.BRIGHT + "═" * 80)
        print(Fore.BLUE + Style.BRIGHT + f" 🚀 WARIANT: {v_name.center(66)} 🚀")
        print(Fore.BLUE + Style.BRIGHT + "═" * 80)

        # Wymiary z pamięci (tylko do logów)
        print(
            Fore.MAGENTA + f"📊 Dane: Train={data['X_train'].shape}, Val={data['X_val'].shape}, Test={data['X_test'].shape}")

        # Przekazuje całą listę trajektorii testowych-
        tester = Tester(test_objects=data["test_obs"], clean_reference=clean_test_obs)
        # Przygotowujemy listę modeli z unikalnymi nazwami dla tego wariantu
        models_to_run = []

        print(f"\n{Fore.YELLOW}🧪 Rozpoczynam testy dla dataset {dataset_name} - {v_name}...")
        for m in models:
            # Dodajemy informację o wariancie do nazwy w tabeli raportu
            models_to_run.append({
                "obj": m["obj"],
                "model_name": f"{m['name']}",
                "dataset_kind": v_name
            })

        tester.run(models_to_run, global_summarizer_dhdt, global_summarizer_h)
else:
    print(f"{Fore.YELLOW}Flaga TEST = {TEST} - brak testów{Style.RESET_ALL}")


# --------------------------------------------------------------------------------------------------------------------
# 8. RAPORT
# --------------------------------------------------------------------------------------------------------------------
if TEST:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'-' * 15} METRYKI DH_DT (pochodne poziomu w zbiornikach) {'-' * 15}")
    # global_summarizer.show_all()
    global_summarizer_dhdt.show_averages()
    global_summarizer_dhdt.save_all_to_file(dataset=dataset_name, save_name_sufix="_dh_dt12")  # Zapis do pliku
    global_summarizer_dhdt.save_averages_to_file(dataset=dataset_name, save_name_sufix="_dh_dt12")

    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'-' * 15} METRYKI H (poziom w zbiornikach){'-' * 15}")
    global_summarizer_h.show_averages()
    global_summarizer_h.save_all_to_file(dataset=dataset_name, save_name_sufix="_h12")  # Zapis do pliku
    global_summarizer_h.save_averages_to_file(dataset=dataset_name, save_name_sufix="_h12")

# --------------------------------------------------------------------------------------------------------------------
# 9. WYKRESY
# --------------------------------------------------------------------------------------------------------------------

if FINAL_PLOTS:
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'═' * 10} 8. GENEROWANIE WYKRESÓW PORÓWNAWCZYCH {'═' * 10}")

    clean_test_obs = reader.find_and_read(dataset_name, "test", noise_level=0.0)

    for var in available_variants:
        v_name = var["name"]  # To jest wariant DANYCH TESTOWYCH (np. CLEAN lub NOISY)
        v_noise = var["noise"]

        print(f"\n{Fore.CYAN}Analiza dla danych testowych: {Style.BRIGHT}{v_name}")
        target_indices = get_unique_signal_indices(dataset_name, mode="test", noise_level=v_noise)
        test_obs = reader.find_and_read(dataset_name, "test", noise_level=v_noise)

        for sig_type, idx in target_indices.items():
            if idx >= len(test_obs): continue

            test_obj = test_obs[idx]
            t_sim, u_sim, h0, dh0 = test_obj.get_data_to_simulate()
            clean_obj = clean_test_obs[idx]
            t_plot, u_plot, h_true_clean, dh_dt_true_clean = clean_obj.get_data_to_plot()

            # --- KLUCZOWA ZMIANA: Grupowanie modeli według wariantu treningowego ---
            # Wyciągamy unikalne warianty treningowe, które mamy w liście models
            training_variants = list(set(m["variant"] for m in models))

            for t_variant in training_variants:
                y_sim_list = []
                dy_sim_list = []
                model_names = []

                # Filtrujemy modele: bierzemy tylko te, które były uczone na t_variant
                filtered_models = [m for m in models if m["variant"] == t_variant]

                for m in filtered_models:
                    model_obj = m["obj"]
                    sim_res = model_obj.simulate(t=t_sim, u_new=u_sim, h0=h0, dh_dt0=dh0)
                    _, _, h_sim, dh_sim = sim_res.get_data_to_plot()

                    y_sim_list.append(h_sim)
                    dy_sim_list.append(dh_sim)
                    # Używamy base_name (np. Torch_LSTM), bo wariant jest już w tytule
                    model_names.append(m["base_name"])

                # Generowanie wykresu dla danej grupy
                print(f"  {Fore.GREEN}└─ Wykres dla: {sig_type.upper()} | Modele uczone na: {t_variant}")

                v_name_safe = str(v_name).replace('.', '_')
                v_name_title = str(v_name).replace('.', ' ')
                t_var_safe = str(t_variant).replace('.', '_')

                SystemPlotter.plot(
                    t=t_plot,
                    u=u_plot,
                    y_true=h_true_clean,
                    dy_dt_true=dh_dt_true_clean,
                    y_sim_list=y_sim_list,
                    dy_dt_sim_list=dy_sim_list,
                    legend_sim=model_names,
                    title=f"Test na: {v_name} | Modele uczone na: {t_variant} | U: {sig_type.upper()}",
                    save_name=f"Test_{idx}_Train_{t_var_safe}_TestData_{v_name_safe}_U_{sig_type.upper()}",
                    dataset=dataset_name,
                    show=show_testing_plot,
                    x_lim=[0, 450]
                )

    print(f"\n{Fore.GREEN}{Style.BRIGHT}✨ Wykresy rozdzielone pomyślnie!")
    plt.show(block=True)
