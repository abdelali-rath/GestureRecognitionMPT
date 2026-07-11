from labeling import data_labeling, dataset_building

print("1 - Gesten prüfen")
print("2 - Datensatz erstellen")

wahl = input("Auswahl: ")

if wahl == "1":
    label = input("Label (A, P, U, I ...): ").upper()
    data_labeling(label)

elif wahl == "2":
    dataset_building("dataset/gesamt_dataset.pkl")

else:
    print("Ungültige Auswahl.")