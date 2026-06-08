# Zhvillimi dhe vlerësimi i një modeli CNN për detektimin e objekteve të spikatura në imazhe

Ky projekt është realizuar si pjesë e punimit të diplomës me temën:

**“Zhvillimi dhe vlerësimi i një modeli CNN për detektimin e objekteve të spikatura në imazhe”**

Qëllimi i projektit është ndërtimi dhe vlerësimi i një modeli të bazuar në **Convolutional Neural Networks (CNN)** për detyrën e **Salient Object Detection**. Modeli merr si hyrje një imazh dhe gjeneron një maskë, ku objekti më i spikatur ndahet nga sfondi.

---

## 1. Përshkrimi i projektit

Në këtë projekt është zhvilluar një model CNN për detektimin e objekteve të spikatura në imazhe. Detyra e modelit është të mësojë nga çiftet imazh–maskë dhe më pas të parashikojë maska për imazhe të reja.

Projekti përfshin këto hapa kryesorë:

* përgatitjen e dataset-it;
* ndarjen e të dhënave në train, validation dhe test;
* ndërtimin e modelit CNN;
* trajnimin e modelit;
* vlerësimin e modelit me metrika të ndryshme;
* ruajtjen e rezultateve vizuale.

Në fund, maskat e parashikuara krahasohen me maskat reale të dataset-it për të vlerësuar performancën e modelit.

---

## 2. Dataset-i i përdorur

Në këtë projekt është përdorur dataset-i **ECSSD (Extended Complex Scene Saliency Dataset)**, i cili përmban imazhe dhe maska për detyrën e detektimit të objekteve të spikatura.

Dataset-i mund të merret nga linku zyrtar:

https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html

Pas shkarkimit të dataset-it, imazhet dhe maskat duhet të vendosen në folderët përkatës brenda projektit.

Struktura e dataset-it duhet të jetë:

```text
dataset/
├── images/
└── masks/
```

Folderi `images/` duhet të përmbajë imazhet origjinale, ndërsa folderi `masks/` duhet të përmbajë maskat reale përkatëse.

---

## 3. Çka është Salient Object Detection?

**Salient Object Detection** është një teknikë në fushën e përpunimit të imazheve dhe inteligjencës artificiale, e cila përdoret për të identifikuar objektin më të spikatur ose më të rëndësishëm në një imazh.

Rezultati zakonisht paraqitet në formë të një maske bardh e zi, ku:

* pjesa e bardhë tregon objektin e spikatur;
* pjesa e zezë tregon sfondin.

Kjo teknikë përdoret në shumë fusha të vizionit kompjuterik, si segmentimi i imazheve, analizimi i skenave dhe përpunimi automatik i imazheve.

---

## 4. Struktura e projektit

Struktura kryesore e projektit është si më poshtë:

```text
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

```text
dataset/
├── images/
└── masks/
```

### `splits/`

Ky folder përmban ndarjen e dataset-it në tri pjesë kryesore:

* train;
* validation;
* test.

Kjo ndarje përdoret që modeli të trajnohet, të kontrollohet gjatë trajnimit dhe më pas të testohet në të dhëna që nuk i ka parë më herët.

### `checkpoints/`

Ky folder përdoret për ruajtjen e modelit pas trajnimit. Modeli i trajnuar ruhet këtu, në mënyrë që të përdoret më vonë për testim ose vlerësim pa pasur nevojë të trajnohet përsëri nga fillimi.

### `results/`

Ky folder përmban rezultatet vizuale të gjeneruara nga modeli. Zakonisht, rezultatet përmbajnë:

* imazhin origjinal;
* maskën reale;
* maskën e parashikuar nga modeli.

### `data_loader.py`

Ky fajll përdoret për leximin dhe përgatitjen e të dhënave. Ai ndihmon në ngarkimin e imazheve dhe maskave, si dhe në përgatitjen e tyre për trajnim.

### `sod_model.py`

Ky fajll përmban arkitekturën e modelit CNN. Këtu definohen shtresat e rrjetit neural, të cilat përdoren për të mësuar karakteristikat e imazheve.

### `train.py`

Ky fajll përdoret për trajnimin e modelit. Gjatë ekzekutimit të tij, modeli mëson nga imazhet dhe maskat e dataset-it.

### `evaluate.py`

Ky fajll përdoret për testimin dhe vlerësimin e modelit. Pas trajnimit, modeli testohet në të dhënat e testimit dhe llogariten metrikat e performancës.

### `utils.py`

Ky fajll përmban funksione ndihmëse që përdoren në pjesë të ndryshme të projektit, si përpunimi i imazheve, ruajtja e rezultateve vizuale dhe llogaritja e metrikave.

### `requirements.txt`

Ky fajll përmban listën e bibliotekave të nevojshme për ekzekutimin e projektit. Përmes tij mund të instalohen të gjitha paketat e nevojshme me një komandë të vetme.

---

## 6. Teknologjitë e përdorura

Në këtë projekt janë përdorur këto teknologji dhe biblioteka:

* Python;
* Visual Studio Code;
* TensorFlow/Keras;
* NumPy;
* Matplotlib;
* OpenCV;
* scikit-learn.

Python është përdorur si gjuhë programuese kryesore, ndërsa Visual Studio Code është përdorur si mjedis për zhvillimin dhe ekzekutimin e projektit.

---

## 7. Instalimi i projektit

Për ta ekzekutuar projektin, fillimisht duhet të jetë i instaluar **Python** në kompjuter.

Pas shkarkimit ose klonimit të repository-t, hapet projekti në Visual Studio Code.

Krijimi i virtual environment bëhet me komandën:

```bash
python -m venv venv
```

Aktivizimi i virtual environment në Windows bëhet me komandën:

```bash
venv\Scripts\activate
```

Pas aktivizimit, instalohen bibliotekat e nevojshme:

```bash
pip install -r requirements.txt
```

---

## 8. Përgatitja e dataset-it

Para trajnimit të modelit, dataset-i duhet të vendoset në strukturën e duhur.

Brenda projektit duhet të krijohet folderi `dataset/`, i cili përmban dy nënfolderë:

```text
dataset/
├── images/
└── masks/
```

Në folderin `images/` vendosen imazhet origjinale të dataset-it, ndërsa në folderin `masks/` vendosen maskat përkatëse.

Shembull:

```text
dataset/
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
│
└── masks/
    ├── 0001.png
    ├── 0002.png
    └── ...
```

Është e rëndësishme që imazhet dhe maskat përkatëse të kenë emra të njëjtë ose të përputhshëm, në mënyrë që të lexohen saktë nga kodi.

---

## 9. Ekzekutimi i projektit

Pas instalimit të bibliotekave dhe përgatitjes së dataset-it, projekti mund të ekzekutohet në dy hapa kryesorë.

### 9.1 Trajnimi i modelit

Për të trajnuar modelin, përdoret komanda:

```bash
python train.py
```

Gjatë këtij procesi, modeli mëson nga dataset-i i përgatitur. Pas përfundimit të trajnimit, modeli ruhet në folderin:

```text
checkpoints/
```

### 9.2 Vlerësimi i modelit

Për të testuar dhe vlerësuar modelin, përdoret komanda:

```bash
python evaluate.py
```

Ky hap përdoret për të parë se sa mirë modeli arrin të parashikojë maskat për imazhet testuese. Rezultatet vizuale ruhen në folderin:

```text
results/
```

---

## 10. Rezultatet e projektit

Rezultatet e projektit paraqiten në formë vizuale dhe numerike.

Rezultatet vizuale zakonisht përmbajnë:

* imazhin origjinal;
* maskën reale;
* maskën e parashikuar nga modeli.

Këto rezultate ndihmojnë për të parë dallimin ndërmjet maskës reale dhe maskës së parashikuar nga modeli.

---

## 11. Metrikat e vlerësimit

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

## 12. Komandat kryesore

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

## 13. Autore

**Rozafa Hajrizi**

Punim diplome
Fakulteti i Inxhinierisë Mekanike dhe Kompjuterike
