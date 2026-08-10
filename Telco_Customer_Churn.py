import pandas as pd

# Učitavamo CSV fajl i čuvamo podatke u DataFrame df.
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Proveravamo broj redova i kolona u dataset-u.
print("Broj redova:", df.shape[0])
print("Broj kolona:", df.shape[1])
print()

# Prikazujemo prvih 5 redova kako bismo dobili osnovni uvid u strukturu i sadržaj podataka.
print(df.head())
print()

# Prikazujemo osnovne informacije o kolonama
df.info()
print()

# Proveravamo da li postoje nedostajuce vrednosti
print("Nedostajuće vrednosti:")
print(df.isnull().sum())
print()

# Proveravamo da li su neke vrednosti predstavljene kao jedan prazan razmak.
print("Vrednosti koje su razmak:")
print((df == " ").sum())
print()

# TotalCharges je u početku učitan kao tekstualna kolona.
# Pokušavamo da sve njene vrednosti pretvorimo u numerički tip.
# Vrednosti koje ne mogu da se konvertuju postaju NaN.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors="coerce")

# Ponovo proveravamo nedostajuće vrednosti nakon konverzije.
print("Nedostajuće vrednosti nakon konverzije TotalCharges:")
print(df.isnull().sum())
print()

# Prikazujemo korisnike kojima nedostaje TotalCharges kako bismo utvrdili prirodu problema.
print("Korisnici sa nedostajućim TotalCharges:")
print(df[df["TotalCharges"].isnull()])
print()

# Uklanjamo redove kojima nedostaje vrednost u TotalCharges.
df = df.dropna(subset=["TotalCharges"])

# Proveravamo da li postoje potpuno duplirani redovi.
print("Broj dupliranih redova:", df.duplicated().sum())

# Proveravamo da li se neki customerID pojavljuje više puta.
print("Broj dupliranih customerID vrednosti:",df["customerID"].duplicated().sum())
print()

# Proveravamo dimenzije dataset-a nakon čišćenja.
print("Broj redova nakon čišćenja:", df.shape[0])
print("Broj kolona:", df.shape[1])
print()

# Proveravamo tipove podataka nakon izvršenog čišćenja.
print("Tipovi podataka nakon čišćenja:")
print(df.dtypes)
print()

# Proveravamo da li su nakon čišćenja ostale neke nedostajuće vrednosti.
print("Nedostajuće vrednosti nakon čišćenja:")
print(df.isnull().sum())
print()

# Prikazujemo osnovnu statistiku za numeričke promenljive.
print("Osnovna statistika numeričkih promenljivih:")
print(df[["SeniorCitizen", "tenure","MonthlyCharges", "TotalCharges"]].describe())
print()

# Proveravamo koje vrednosti postoje u SeniorCitizen i koliko korisnika pripada svakoj kategoriji.
print("Raspodela SeniorCitizen:")
print(df["SeniorCitizen"].value_counts())
print()

# Izdvajamo sve kolone koje imaju tekstualne vrednosti.
categorical_columns = df.select_dtypes(include="object").columns

# Za svaku kategorijsku promenljivu prikazujemo sve postojeće kategorije i broj korisnika u svakoj kategoriji.
for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts())