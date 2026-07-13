from labeling import data_labeling, dataset_building
from hmmclassifier import train_and_evaluate

print("1 - Gesten prüfen")
print("2 - Datensatz erstellen")
print("3 - HMM trainieren")

wahl = input("Auswahl: ")

if wahl == "1":
    label = input("Label (A, P, U, I ...): ").upper()
    data_labeling(label)

elif wahl == "2":
    dataset_building("dataset/gesamt_dataset.pkl")

elif wahl == "3":
    train_and_evaluate()

else:
    print("Ungültige Auswahl.")
