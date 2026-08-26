# Telco Customer Churn

Seminarski rad iz predmeta Uvod u nauku o podacima — predviđanje odliva
korisnika (customer churn) telekomunikacione kompanije.

## O projektu

Cilj projekta je da se na osnovu karakteristika korisnika, njegovog ugovora, korišćenih usluga i troškova predvidi da li će korisnik napustiti kompaniju (`Churn`).

Projekat prati kompletan tok jednog Data Science projekta:

**definisanje problema → pregled i čišćenje podataka → provera kvaliteta → podela na train/test → detekcija anomalija → feature engineering → EDA i statistička analiza → modeliranje → evaluacija → cross-validacija → tuning hiperparametara → izbor modela → poslovne preporuke**

Posebna pažnja posvećena je sprečavanju *data leakage*-a: odluke o transformaciji i pripremi podataka donose se na trening skupu, dok se test skup čuva za nezavisnu finalnu evaluaciju.

---

## Cilj analize

Problem je formulisan kao **binarna klasifikacija**:

- `0` — korisnik ostaje
- `1` — korisnik odlazi

Glavni cilj nije samo ostvariti visoku Accuracy metriku, već što pouzdanije prepoznati korisnike koji će zaista otići. Zbog toga su posebno važne metrike **Recall**, **F1-score** i **ROC-AUC**.

---

## Skup podataka

Korišćen je skup podataka **Telco Customer Churn**, koji sadrži informacije o korisnicima telekomunikacione kompanije, uključujući:

- demografske karakteristike,
- dužinu korišćenja usluge (`tenure`),
- tip ugovora (`Contract`),
- vrstu internet usluge (`InternetService`),
- dodatne usluge kao što su `OnlineSecurity` i `TechSupport`,
- način plaćanja (`PaymentMethod`),
- mesečne i ukupne troškove (`MonthlyCharges`, `TotalCharges`),
- ciljnu promenljivu `Churn`.

Dataset je javno dostupan i potiče iz IBM Telco Customer Churn primera, a korišćena verzija je preuzeta sa Kaggle-a.
Skup podataka: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle)

### Osnovne karakteristike

- približno **7.000 korisnika**
- ciljna promenljiva: `Churn`
- klase su nebalansirane:
  - oko **73%** korisnika ostaje
  - oko **27%** korisnika odlazi

Tokom inicijalnog pregleda uočeno je da `TotalCharges` sadrži 11 praznih vrednosti. One odgovaraju korisnicima sa `tenure = 0`, pa se problem obrađuje tokom pripreme podataka.

---

## Tok projekta

Projekat je organizovan kroz devet Jupyter notebook sveski:

| Notebook | Opis |
|---|---|
| `00_definisanje_problema.ipynb` | Definisanje problema, cilja i konteksta analize |
| `01_inicijalni_pregled.ipynb` | Dimenzije skupa, tipovi podataka i početna kontrola kvaliteta |
| `02_sistematske_greske.ipynb` | Provera sistematskih i logičkih grešaka između povezanih atributa |
| `03_train_test_split.ipynb` | Stratifikovana podela podataka na trening i test skup |
| `04_detekcija_anomalija.ipynb` | Detekcija statističkih anomalija pomoću IQR metode |
| `05_priprema_podataka.ipynb` | Čišćenje, feature engineering, enkodiranje i priprema za ML |
| `06_iterativna_eda.ipynb` | Eksplorativna analiza i statističko testiranje hipoteza |
| `07_modeliranje_i_evaluacija.ipynb` | Treniranje, evaluacija, validacija i optimizacija modela |
| `08_zakljucak.ipynb` | Objedinjeni zaključci i poslovne preporuke |

Dodatno je dostupan i objedinjeni HTML dokument `kompletan_seminarski.html`.

---

## Priprema i čišćenje podataka

U fazi pripreme podataka urađeno je:

1. provera strukture i tipova podataka,
2. identifikacija nedostajućih vrednosti,
3. obrada `TotalCharges`,
4. provera sistematskih grešaka između povezanih kolona,
5. detekcija anomalija pomoću **IQR** metode,
6. uklanjanje identifikacionog atributa `customerID`,
7. izdvajanje ciljne promenljive `Churn`,
8. kreiranje novih atributa,
9. enkodiranje kategorijskih promenljivih,
10. standardizacija numeričkih atributa tamo gde je potrebna,
11. priprema odvojenih skupova za mašinsko učenje.

### Feature engineering

Kreirana su dva značajna nova atributa:

- **`ActiveServices`** — ukupan broj aktivnih dodatnih usluga korisnika.
- **`TenureGroup`** — grupisana dužina korišćenja usluge:
  - `0–12`
  - `13–24`
  - `25–48`
  - `49+` meseci.

`ActiveServices` se pokazao kao koristan prediktor, naročito kod modela zasnovanih na stablima.

---
### Provera multikolinearnosti

Nakon uočene visoke Spearman korelacije između `tenure` i `TotalCharges`
(~0.89), sprovedena je formalna provera pomoću **VIF (Variance Inflation
Factor)** analize. Rezultati su pokazali:

- `tenure`: VIF ≈ 41.5 (umerena multikolinearnost)
- `MonthlyCharges`: VIF ≈ 3.4 (bez problema)
- `TotalCharges`: VIF ≈ 8.2 (umerena multikolinearnost)
- `TenureGroup`: VIF > 20 (visoka multikolinearnost sa `tenure`)

Na osnovu ovoga doneta je odluka:

- **Za linearne modele** (Logistička regresija): izbacuju se `TotalCharges`
  i `TenureGroup` pre treniranja, radi stabilnijih koeficijenata i lakše
  interpretacije.
- **Za modele bazirane na stablima** (Random Forest, XGBoost): zadržavaju
  se sve kolone jer stabla ne pate od multikolinearnosti.

## Eksplorativna analiza i statističko testiranje

EDA je korišćena kako bi se razumeli obrasci u podacima i identifikovali potencijalno važni prediktori.

Za numeričke promenljive proverena je normalnost raspodele. Pošto raspodele nisu zadovoljile pretpostavku normalnosti, za poređenje grupa korišćen je:

- **Mann–Whitney U test**

Za kategorijske promenljive u odnosu na `Churn` korišćen je:

- **Hi-kvadrat test nezavisnosti**

Analiza je pokazala statistički značajne veze između churn-a i više karakteristika korisnika.

### Multivarijantna analiza

Pored bivarijantne analize svake promenljive u odnosu na `Churn`, sprovedena
je i multivarijantna analiza kroz tri prikaza:

- **Heatmap `Contract × TenureGroup` vs Churn %** — pokazuje da kombinacija
  `Month-to-month` ugovora sa kratkim stažom (0-12 meseci) daje churn
  oko 55%, dok ista dužina staža sa `Two year` ugovorom daje ispod 5%.
- **Grouped bar `Contract × InternetService`** — otkriva da najveći
  churn imaju korisnici sa Fiber optic internetom na Month-to-month
  ugovoru (~55%).
- **Scatter `MonthlyCharges vs tenure` obojeno po Churn** — jasno se
  izdvaja "high-risk zona": novi korisnici (< 20 meseci) sa visokim
  mesečnim troškovima (> 70$).

Ključan zaključak: efekti atributa se **kombinuju multiplikativno**, ne
aditivno. Ovi obrasci se ne mogu videti u bivarijantnoj analizi.

### Najvažniji nalazi

- `Contract` je jedan od najjačih prediktora.
- Korisnici sa **Month-to-month** ugovorom imaju znatno veći churn.
- Kraći `tenure` je povezan sa većim rizikom odlaska.
- Korisnici sa **Fiber optic** internetom imaju viši churn od korisnika sa DSL uslugom.
- Nedostatak dodatnih usluga poput `OnlineSecurity` i `TechSupport` povezan je sa većim churn-om.
- `PaymentMethod` i `PaperlessBilling` takođe pokazuju značajne veze sa odlaskom korisnika.
- `tenure`, `MonthlyCharges` i `TotalCharges` su međusobno snažno povezani, što je važno uzeti u obzir kod interpretacije linearnih modela.

Važno: statistička povezanost ne znači automatski i uzročnost.

---

## Modeli mašinskog učenja

Testirana su tri modela:

### 1. Logistička regresija

Koristi se kao interpretabilan osnovni model za binarnu klasifikaciju.

Prednosti:

- jednostavna interpretacija,
- koeficijenti pokazuju smer i relativni uticaj atributa,
- dobra performansa na problemima binarne klasifikacije.

Zbog nebalansiranih klasa korišćen je `class_weight="balanced"`, a numeričke promenljive su standardizovane.

### 2. Random Forest

Ansambl više stabala odlučivanja koji može da modeluje nelinearne odnose između atributa.

Korišćen je `class_weight="balanced"` kako bi se uzela u obzir nebalansiranost ciljnih klasa.

### 3. XGBoost

Napredni boosting algoritam kod kojeg se stabla grade sekvencijalno, pri čemu svako naredno stablo pokušava da ispravi greške prethodnih.

Za nebalansirane klase korišćen je `scale_pos_weight`.

---

## Evaluacija modela

Modeli su poređeni pomoću sledećih metrika:

- **Accuracy** — procenat svih tačnih predikcija.
- **Precision** — koliko je predviđenih odlazaka zaista bilo odlazak.
- **Recall** — koliko je stvarnih odlazaka model uspeo da prepozna.
- **F1-score** — harmonijska sredina Precision i Recall metrike.
- **ROC-AUC** — sposobnost modela da razlikuje dve klase za različite pragove odlučivanja.

Kod ovog problema **Recall ima posebno veliki značaj**, jer je poslovno skuplje propustiti korisnika koji će zaista otići nego kontaktirati korisnika koji ipak ostaje.

### Rezultati početnih modela

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistička regresija | 72.42% | 48.84% | **78.88%** | 60.33% | **83.42%** |
| Random Forest | **77.19%** | **55.98%** | 66.31% | 60.71% | 82.15% |
| XGBoost | 73.92% | 50.61% | 77.81% | **61.33%** | 83.01% |

Rezultati pokazuju da nijedan model nije najbolji po svim metrikama:

- Random Forest ima najbolju Accuracy i Precision.
- Logistička regresija ima najbolji Recall i ROC-AUC među početnim modelima.
- XGBoost ima najbolji F1-score.

---

## Cross-validation

Za proveru stabilnosti rezultata primenjena je **Stratified 5-Fold Cross-Validation**.

Cross-validacija se izvršava isključivo na trening skupu, dok test skup ostaje nezavisan i koristi se tek za finalnu evaluaciju.

Na ovaj način proverava se da li performanse modela zavise previše od jedne konkretne train/test podele.

---

## Podešavanje hiperparametara

Za modele koji su se najbolje pokazali primenjene su sistematske metode pretrage:

- **GridSearchCV** za logističku regresiju,
- **RandomizedSearchCV** za XGBoost.

Pretraga je sprovedena na trening skupu uz stratifikovanu 5-struku cross-validaciju, a kao glavna optimizaciona metrika korišćen je **F1-score**.

### Rezultati nakon tuninga

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistička regresija | 72.49% | 48.92% | 78.61% | 60.31% | 83.47% |
| XGBoost | 73.85% | 50.50% | **80.48%** | **62.06%** | **84.17%** |

Podešavanje hiperparametara je posebno poboljšalo XGBoost, koji nakon tuninga ostvaruje najbolje rezultate po F1-score-u i ROC-AUC-u među testiranim finalnim konfiguracijama.

---

## Podešavanje praga odlučivanja

Podrazumevani prag od `0.5` nije nužno optimalan za churn problem.

Analizirano je kako promena praga utiče na:

- Precision,
- Recall,
- F1-score.

Spuštanjem praga povećava se Recall, jer model označava veći broj korisnika kao potencijalno rizične. Istovremeno opada Precision zbog većeg broja False Positive predikcija.

Za poslovni scenario u kojem je prioritet da se propusti što manje korisnika koji odlaze, prag oko **0.3–0.4** predstavlja razuman kompromis.

---

## Feature Importance

Analizirana je važnost atributa kod sva tri modela.

Među najvažnijim atributima izdvajaju se:

- `Contract`
- `tenure`
- `MonthlyCharges`

`ActiveServices` se takođe pokazao kao koristan novi atribut, posebno kod Random Forest i XGBoost modela.

Pored ugrađenih mera važnosti korišćena je i **permutaciona važnost**, čime je omogućeno pravednije poređenje doprinosa atributa na test skupu.

---

## Konačna preporuka

Za praktičnu primenu izbor modela zavisi od poslovnog prioriteta.

Ako je najvažnije **uhvatiti što veći broj korisnika koji će otići**, Recall treba da ima prioritet.

Na osnovu finalnih rezultata, **podešeni XGBoost** predstavlja najjaču ukupnu opciju u ovom eksperimentu:

- Recall: **80.48%**
- F1-score: **62.06%**
- ROC-AUC: **84.17%**

Logistička regresija ostaje veoma vredna alternativa zbog jednostavnosti i interpretabilnosti, posebno kada je potrebno jasno objasniti zašto je određeni korisnik označen kao rizičan.

---

## Poslovne preporuke

Na osnovu analize mogu se izdvojiti sledeće preporuke:

1. **Prioritetno raditi sa korisnicima koji imaju Month-to-month ugovor**, uz promocije koje ih podstiču da pređu na dugoročnije ugovore.
2. **Posebnu pažnju posvetiti prvih 6–12 meseci**, kada je rizik odlaska najveći.
3. **Podsticati korišćenje dodatnih usluga**, naročito `TechSupport` i `OnlineSecurity`.
4. **Detaljnije analizirati Fiber optic korisnike**, jer imaju neuobičajeno visok churn i potrebno je utvrditi da li su uzrok cena, kvalitet usluge ili konkurencija.
5. Koristiti model za **proaktivnu identifikaciju rizičnih korisnika** i ciljane retention kampanje.
6. Prag odlučivanja prilagoditi poslovnom trošku False Negative i False Positive predikcija.

---

## Ograničenja

Rezultate treba posmatrati u kontekstu korišćenog dataseta.

Glavna ograničenja su:

- performanse modela su solidne, ali postoji prostor za poboljšanje,
- klase su nebalansirane,
- korelacija atributa ne predstavlja dokaz uzročnosti,
- `tenure` i `TotalCharges` su snažno povezani,
- korišćen je jedan konkretan dataset, pa generalizaciju na druge telekomunikacione kompanije treba dodatno proveriti.

Mogući pravci budućeg rada uključuju:

- SMOTE ili druge metode balansiranja,
- širu pretragu hiperparametara,
- dodatni feature engineering,
- naprednije modele,
- kalibraciju verovatnoća,
- testiranje na novijem ili eksternom skupu podataka,
- eksperimentalnu proveru poslovnih intervencija.

---

## Tehnologije i biblioteke

Projekat je realizovan u Python/Jupyter okruženju.

Korišćene biblioteke uključuju:

- Python
- Jupyter Notebook
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- XGBoost

---


## Pokretanje projekta

### 1. Kloniranje repozitorijuma

```bash
git clone <URL_REPOZITORIJUMA>
cd Telco-Customer-Churn
```

### 2. Instalacija potrebnih biblioteka

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost jupyter
```

### 3. Pokretanje Jupyter Notebook-a

```bash
jupyter notebook
```

ili:

```bash
jupyter lab
```

### 4. Redosled izvršavanja

Notebook sveske treba pokretati redom:

```text
00 → 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08
```

Ovakav redosled je važan zato što kasnije sveske koriste podatke i rezultate generisane u prethodnim fazama.

---

## Zaključak

Projekat pokazuje kompletan proces rešavanja problema predikcije churn-a, od inicijalnog pregleda sirovih podataka do izgradnje i evaluacije modela mašinskog učenja.

Analiza pokazuje da je odlazak korisnika predvidiv u razumnoj meri, sa ROC-AUC vrednošću od približno **0.84** kod najbolje podešene konfiguracije. Najvažniji faktori uključuju tip ugovora, dužinu korišćenja usluge, mesečne troškove i korišćenje dodatnih usluga.

Konačni rezultat nije samo model sa određenom metrikom, već analitički proces koji povezuje kvalitet podataka, statističku analizu, mašinsko učenje i konkretne poslovne preporuke za smanjenje odliva korisnika.


## Tim

Jana Pavlovic 91/2023, Mihailo Smiljic 96/2023
