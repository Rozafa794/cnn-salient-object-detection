# Zhvillimi dhe vlerësimi i një modeli CNN për Salient Object Detection

Ky projekt është realizuar si pjesë e punimit të diplomës me temën:

**“Zhvillimi dhe vlerësimi i një modeli CNN për Salient Object Detection”**

Qëllimi i projektit është ndërtimi dhe vlerësimi i një modeli të bazuar në **Convolutional Neural Networks (CNN)**, i cili përdoret për detektimin e objekteve të spikatura në imazhe.

Me fjalë më të thjeshta, modeli merr një imazh si hyrje dhe përpiqet të gjejë objektin më të rëndësishëm ose më të dukshëm në atë imazh. Rezultati i modelit është një maskë, ku pjesa kryesore e objektit paraqitet ndaras nga sfondi.

---

## 1. Përshkrimi i projektit

Në këtë projekt është zhvilluar një model CNN për detyrën e **Salient Object Detection**. Kjo detyrë ka për qëllim identifikimin e pjesëve më të rëndësishme në një imazh.

Projekti përfshin disa hapa kryesorë:

* përgatitjen e dataset-it;
* ndarjen e të dhënave në train, validation dhe test;
* ndërtimin e modelit CNN;
* trajnimin e modelit;
* vlerësimin e modelit me metrika të ndryshme;
* ruajtjen e rezultateve vizuale.

Në fund, modeli gjeneron maska të parashikuara, të cilat krahasohen me maskat reale të dataset-it.

---

## 2. Çka është Salient Object Detection?

**Salient Object Detection** është një teknikë në fushën e përpunimit të imazheve dhe inteligjencës artificiale, e cila përdoret për të gjetur objektin më të spikatur në një imazh.

Për shembull, nëse një imazh përmban një kafshë, një person ose një objekt të caktuar në qendër të vëmendjes, modeli tenton ta dallojë atë objekt nga sfondi.

Rezultati zakonisht paraqitet në formë të një **maske bardh e zi**, ku:

* pjesa e bardhë tregon objektin e spikatur;
* pjesa e zezë tregon sfondin.

---

## 3. Qëllimi i projektit

Qëllimi kryesor i këtij projekti është të zhvillohet një model CNN që mund të mësojë nga imazhet dhe maskat përkatëse, në mënyrë që më pas të parashikojë maska të reja për imazhe të panjohura.

Objektivat kryesore të projektit janë:

* krijimi i një modeli CNN për detektimin e objekteve të spikatura;
* përdorimi i një dataset-i me imazhe dhe maska;
* trajnimi i modelit me të dhëna të përgatitura;
* testimi i modelit në imazhe të ndara për vlerësim;
* krahasimi i maskës reale me maskën e parashikuar;
* vlerësimi i performancës me metrika si Precision, Recall, F1-score dhe IoU.

---

## 4. Struktura e projektit

Struktura e projektit është organizuar në folderë dhe fajlla të ndryshëm. Secili prej tyre ka një rol të veçantë.

```bash
project/
│
├── dataset/
│   ├── images/
│   └── masks/
│
├── splits/
│
├── checkpoints/
│
├── results/
│
├── data_loader.py
├── sod_model.py
├── train.py
├── evaluate.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## 5. Përshkrimi i folderëve dhe fajllave

### `dataset/`

Ky folder përmban dataset-in e përdorur në projekt. Brenda tij ruhen imazhet origjinale dhe maskat përkatëse.

Zakonisht struktura është:

```bash
dataset/
├── images/
└── masks/
```

Folderi `images/` përmban imazhet origjinale, ndërsa folderi `masks/` përmban maskat reale të atyre imazheve.

---

### `splits/`

Ky folder përmban ndarjen e dataset-it në tri pjesë kryesore:

* train;
* validation;
* test.

Kjo ndarje përdoret që modeli të trajnohet, të kontrollohet gjatë trajnimit dhe më pas të testohet në të dhëna që nuk i ka parë më herët.

---

### `checkpoints/`

Ky folder përdoret për ruajtjen e modelit pas trajnimit.

Modeli i trajnuar ruhet në këtë folder, në mënyrë që të përdoret më vonë për testim ose vlerësim pa pasur nevojë të trajnohet përsëri nga fillimi.

---

### `results/`

Ky folder përmban rezultatet vizuale të gjeneruara nga modeli.

Në këtë folder ruhen imazhet ku zakonisht paraqiten:

* imazhi origjinal;
* maska reale;
* maska e parashikuar nga modeli.

---

### `data_loader.py`

Ky fajll përdoret për leximin dhe përgatitjen e të dhënave.

Ai ndihmon në ngarkimin e imazheve dhe maskave, si dhe në përgatitjen e tyre për trajnim.

---

### `sod_model.py`

Ky fajll përmban arkitekturën e modelit CNN.

Këtu definohen shtresat e rrjetit neural, të cilat përdoren për të mësuar karakteristikat e imazheve.

---

### `train.py`

Ky fajll përdoret për trajnimin e modelit.

Gjatë ekzekutimit të këtij fajlli, modeli mëson nga imazhet dhe maskat e dataset-it.

---

### `evaluate.py`

Ky fajll përdoret për testimin dhe vlerësimin e modelit.

Pas trajnimit, modeli testohet në të dhënat e testimit dhe llogariten metrikat e performancës.

---

### `utils.py`

Ky fajll përmban funksione ndihmëse që përdoren në pjesë të ndryshme të projektit.

Këto funksione mund të përdoren për përpunimin e imazheve, ruajtjen e rezultateve ose llogaritjen e metrikave.

---

### `requirements.txt`

Ky fajll përmban listën e bibliotekave të nevojshme për ekzekutimin e projektit.

Me anë të këtij fajlli mund të instalohen të gjitha bibliotekat e nevojshme me një komandë të vetme.

---

## 6. Teknologjitë e përdorura

Në këtë projekt janë përdorur këto teknologji dhe biblioteka:

* Python;
* Visual Studio Code;
* PyTorch ose TensorFlow;
* NumPy;
* Matplotlib;
* OpenCV;
* Scikit-learn.

Python është përdorur si gjuhë programuese kryesore, ndërsa Visual Studio Code është përdorur si mjedis për zhvillimin e projektit.

---

## 7. Instalimi i projektit

Për ta ekzekutuar projektin, fillimisht duhet të siguroheni që në kompjuter është i instaluar Python.

Pas kësaj, hapet projekti në Visual Studio Code dhe krijohet një virtual environment.

Komanda për krijimin e virtual environment është:

```bash
python -m venv venv
```

Pastaj aktivizohet virtual environment.

Në Windows përdoret komanda:

```bash
venv\Scripts\activate
```

Pas aktivizimit, instalohen bibliotekat e nevojshme me komandën:

```bash
pip install -r requirements.txt
```

---

## 8. Ekzekutimi i projektit

Pas instalimit të bibliotekave, projekti mund të ekzekutohet në dy hapa kryesorë.

### 8.1 Trajnimi i modelit

Për të trajnuar modelin, ekzekutohet komanda:

```bash
python train.py
```

Gjatë këtij procesi, modeli mëson nga dataset-i i përgatitur.

Pas përfundimit të trajnimit, modeli ruhet në folderin:

```bash
checkpoints/
```

---

### 8.2 Vlerësimi i modelit

Për të testuar dhe vlerësuar modelin, ekzekutohet komanda:

```bash
python evaluate.py
```

Ky hap përdoret për të parë se sa mirë modeli arrin të parashikojë maskat për imazhet testuese.

Rezultatet vizuale ruhen në folderin:

```bash
results/
```

---

## 9. Rezultatet e projektit

Rezultatet e projektit paraqiten në formë vizuale dhe numerike.

Rezultatet vizuale zakonisht përmbajnë tri pjesë:

1. imazhin origjinal;
2. maskën reale;
3. maskën e parashikuar nga modeli.

Këto rezultate ndihmojnë për të parë në mënyrë më të qartë dallimin ndërmjet asaj që modeli ka parashikuar dhe maskës së saktë.

---

## 10. Metrikat e vlerësimit

Për vlerësimin e modelit përdoren disa metrika kryesore.

### Precision

Precision tregon sa nga pikselët që modeli i ka parashikuar si objekt të spikatur janë vërtet pjesë e objektit.

### Recall

Recall tregon sa mirë modeli arrin t’i gjejë pikselët që realisht i përkasin objektit të spikatur.

### F1-score

F1-score është një kombinim ndërmjet Precision dhe Recall. Kjo metrikë përdoret për të dhënë një vlerësim më të balancuar të performancës së modelit.

### IoU

IoU, ose Intersection over Union, tregon sa përputhet maska e parashikuar me maskën reale.

Sa më e lartë të jetë vlera e IoU, aq më e mirë është përputhja ndërmjet maskës reale dhe maskës së parashikuar.

---

## 11. Si funksionon projekti në mënyrë të thjeshtë?

Procesi i punës së projektit mund të përshkruhet kështu:

1. Lexohen imazhet dhe maskat nga dataset-i.
2. Imazhet përgatiten për modelin.
3. Modeli CNN trajnohet duke përdorur imazhet dhe maskat reale.
4. Pas trajnimit, modeli testohet në imazhe të reja.
5. Modeli gjeneron maska të parashikuara.
6. Maskat e parashikuara krahasohen me maskat reale.
7. Llogariten metrikat e performancës.
8. Rezultatet ruhen në folderin `results/`.

---

## 12. Shembull i rezultatit

Një rezultat i gjeneruar nga modeli përmban krahasimin ndërmjet:

* imazhit origjinal;
* maskës reale;
* maskës së parashikuar.

Kjo ndihmon për të kuptuar nëse modeli ka arritur ta dallojë saktë objektin e spikatur në imazh.

---

## 13. Kërkesat për ekzekutim

Për të ekzekutuar projektin nevojiten:

* Python i instaluar;
* Visual Studio Code ose ndonjë editor tjetër;
* bibliotekat e listuara në `requirements.txt`;
* dataset-i me imazhe dhe maska;
* strukturë e saktë e folderëve.

---

## 14. Komandat kryesore

Krijimi i virtual environment:

```bash
python -m venv venv
```

Aktivizimi në Windows:

```bash
venv\Scripts\activate
```

Instalimi i bibliotekave:

```bash
pip install -r requirements.txt
```

Trajnimi i modelit:

```bash
python train.py
```

Vlerësimi i modelit:

```bash
python evaluate.py
```


---

## 16. Autore

**Rozafa Hajrizi**

Punim diplome
Fakulteti i Inxhinierisë Mekanike dhe Kompjuterike
