# Studio: predizione della morte cardiovascolare a 7 anni (pazienti IHD)


## 0. Ruolo e obiettivo

Agisci come un data scientist clinico senior. Devi costruire una pipeline riproducibile che classifichi la morte cardiovascolare entro 7 anni, svolga survival analysis, quantifichi se i parametri tiroidei aggiungono valore prognostico ai marker cardiaci, produca grafici e una SHAP analysis. Lavora in modo incrementale (l'ambiente può avere 1 solo core e timeout brevi: rendi ogni script *ripristinabile*, saltando le celle già calcolate).

## 1. Difetto da correggere (contesto critico)

La tesi originale usava il flag grezzo `CVD Death` come target su tutti i pazienti, dichiarando "87% sopravvive a 7 anni". È un errore: quell'87% è solo "nessuna morte CVD registrata" e **ignora la censura a destra**. Pazienti con follow-up < 7 anni (vivi all'ultimo contatto) o morti per cause non-CVD prima dei 7 anni hanno uno **stato a 7 anni non osservato** e non vanno etichettati come sopravvissuti. Devi gestire la censura esplicitamente.

## 2. Caricamento e merge (attenzione ai nomi-colonna con newline, ripuliscili)

- `raw_data.xlsx`: 8144 righe, 83 colonne. **Tieni solo le righe con `Number` non nullo → 8065 pazienti.** Cast `Number` a int.
- `data_prelievo.xlsx`: `Number`, `Data prelievo` (= tempo zero / arruolamento).
- `creatina_more_columns.xlsx`: `Number`, `Total cholesterol`, `HDL`, `LDL`, `Triglycerides`, `Creatinina`.
- Merge su `Number` (left).
- **Diverse colonne di `raw_data` contengono un newline `\n` nel nome.** Usa ESATTAMENTE questi nomi grezzi nel rename:

```python
RENAME = {
 'Gender (Male = 1)':'Gender','Age':'Age','Angina':'Angina',
 'Previous CABG':'Previous_CABG','Previous PCI':'Previous_PCI',
 'Previous Myocardial Infarction':'Previous_MI','Acute Myocardial Infarction':'Acute_MI',
 'Angiography':'Angiography','Vessels':'Vessels',
 'Documented resting \nor exertional ischemia':'Ischemia',
 'Post-ischemic Dilated\nCardiomyopathy':'PostIsch_DCM',
 'Smoke\nHistory of smoke':'Smoke','Diabetes\nHistory of diabetes':'Diabetes',
 'Hypertension\nHistory of hypertension':'Hypertension',
 'Dyslipidemia\nHystory of dyslipidemia':'Dyslipidemia',
 'Atrial Fibrillation':'AFib','fe':'fe',
 'TSH':'TSH','fT3':'fT3','fT4':'fT4','Euthyroid':'Euthyroid',
 'Subclinical primary hypothyroidism (SCH)':'SCH',
 'Subclinical primary hyperthyroidism\n(SCT)':'SCT',
 'Low T3':'Low_T3','Ipotiroidismo':'Hypothyroid','Ipertiroidismo':'Hyperthyroid'}
```

Colonne date da parsare: `Data prelievo`, `Follow Up Data`, `Data of death`. Esiti grezzi disponibili: `Total mortality`, `CVD Death`, `Fatal MI or Sudden death`, `Cause of death`.

## 3. Tempo-evento, eventi, target

```
t_end      = Data of death se presente, altrimenti Follow Up Data
time_days  = (t_end - Data prelievo).days, troncato a >= 0
time_years = time_days / 365.25
event_cvd     = (CVD Death == 1)
event_death   = (Total mortality == 1)
event_noncvd  = event_death & ~event_cvd
HORIZON_DAYS  = 7 * 365.25
```

Target di classificazione **"morte CVD entro 7 anni" (y7)**:
- `y7 = 1` se `event_cvd & time_days <= HORIZON_DAYS` (843 pazienti)
- `y7 = 0` se `time_days >= HORIZON_DAYS & non(cvd_within)` → event-free a 7 anni (3547)
- altrimenti **censurato → da rimuovere** (3675: 2629 vivi con FU<7a + 1046 morti non-CVD <7a)

Produci **DUE coorti di classificazione**:
- **strict** (cause-specific): solo {event, eventfree} → **N=4390, positivi 843 = 19,2%**. I decessi non-CVD <7a sono rimossi.
- **competing**: strict + (decessi non-CVD <7a come **negativi**, "non sono morti di CVD entro 7a") → **N=5436, positivi 843 = 15,5%**.

Per la **survival analysis** usa **tutti i pazienti con `time_days > 0`** (tutti i pazienti), evento = `event_cvd` (1007 eventi, 7055 censurati). Qui i censurati NON si rimuovono: sono il punto di forza dell'analisi.

## 4. Feature engineering e feature set

- 17 cardiovascolari: `CARDIO_17 = [Gender, Age, Angina, Previous_CABG, Previous_PCI, Previous_MI, Acute_MI, Angiography, Vessels, Ischemia, PostIsch_DCM, Smoke, Diabetes, Hypertension, Dyslipidemia, AFib, fe]`
- 9 tiroidei: `THYROID_RAW9 = [TSH, fT3, fT4, Euthyroid, SCH, SCT, Low_T3, Hypothyroid, Hyperthyroid]`
- Continui da standardizzare (StandardScaler): `Age, Vessels, fe, TSH, fT3, fT4, fT3_fT4_ratio`
- Derivate: `fT3_fT4_ratio = fT3/fT4`; `Thyroid_abnormal = (Euthyroid==0)`; `thyroid_state` (categoria singola; i 6 stati sono mutuamente esclusivi); `thyroid_ord` = mappa ordinale `{Hypothyroid:-2, SCH:-1, Euthyroid:0, Low_T3:0, SCT:1, Hyperthyroid:2}` — **NB: `Low_T3` (1375 pz) non sta sull'asse ipo↔iper, trattalo come neutro (0)**.

Otto feature set:
```
CV17            = CARDIO_17
CV17_THY26      = CARDIO_17 + THYROID_RAW9
CV17_BIN        = CARDIO_17 + [Thyroid_abnormal]      # cardio + BINARIA eutiroideo/non-eutiroideo
CV17_ORD        = CARDIO_17 + [thyroid_ord]            # cardio + ORDINALE ipo->iper
CV17_CONT       = CARDIO_17 + [TSH, fT3, fT4]
CV17_CAT        = CARDIO_17 + [Euthyroid, SCH, SCT, Low_T3, Hypothyroid, Hyperthyroid]  # cardio + ONE-HOT categorie separate
CV17_RATIO      = CARDIO_17 + THYROID_RAW9 + [fT3_fT4_ratio]
CV17_RATIO_ONLY = CARDIO_17 + [TSH, fT3, fT4, fT3_fT4_ratio]
```
Ognuna delle seguenti richieste va ripetuta per **ciascuno degli 8 set**, in modo da verificare se qualcuno migliora i risultati.

## 5. Analisi esplorativa (EDA) + grafici

1. **Cohort flow** (diagramma/numeri): 8065 → split in event / eventfree / censored, con sottotipi di censura.
2. **Missingness**: verifica (atteso: nessun missing nelle 26 feature della coorte pulita; la creatinina è l'unica con buchi → esclusa).
3. **Associazione univariata col target** (coorte strict): per i continui Mann-Whitney U + rank-biserial; per i binari odds ratio + χ². Ordina per effetto.
4. **GRAFICO 1** — istogrammi/violini delle feature continue (Age, Vessels, fe, TSH, fT3, fT4) **per classe** del target.
5. **GRAFICO 2** — clustering gerarchico delle 26 feature su distanza `1-|Spearman|` (dendrogramma) + **GRAFICO 3** heatmap di correlazione Spearman ordinata per cluster.
6. **Clustering dei pazienti**: KMeans su feature standardizzate, scegli k via silhouette per k=2..6; **GRAFICO 4** proiezione PCA (2D) colorata per cluster e per target. Atteso: silhouette ≈ 0,11 (nessuna struttura forte) → riportalo come risultato onesto.

## 6. Classificazione (massimizza F1-macro)

Pipeline `imblearn.Pipeline([StandardScaler, sampler, classifier])` (il sampler **dentro** la CV, mai sull'intero dataset → no leakage).

- **Modelli (8)**: LogisticRegression, SVC(rbf), KNeighbors, RandomForest, AdaBoost, HistGradientBoosting, XGBoost, MLP.
- **Sampler (5)**: none, RandomUnderSampler, SMOTE, BorderlineSMOTE, SVMSMOTE.
- **Screening** (full grid): 2 coorti × 8 feature set × 8 modelli × 5 sampler, su split stratificato 70/30, F1-macro su test (+ F1 per classe, ROC-AUC, PR-AUC, precision/recall). Salva ogni riga su `results_clf.csv` (append, ripristinabile).
- **CV robusta** sui set/modelli forti (LR, RF, HistGB, XGB con SMOTE): StratifiedKFold 5-fold, riportando media±sd di F1-macro e AUC. **Ottimizza la soglia decisionale** per max F1-macro (cerca su `np.linspace(0.1,0.9,41)` usando le predizioni sul training fold) — leva chiave assente nell'impostazione originale.
- **Fine-tuning** dei leader: `RandomizedSearchCV(scoring='f1_macro', cv=4, n_iter≈15)` + soglia + SVMSMOTE su almeno {CV17, CV17_THY26, CV17_CONT}.
- **GRAFICO 5** — AUC (5-fold CV) per feature set, una linea per modello, due pannelli (strict/competing). Deve mostrare che le 8 curve sono quasi sovrapposte.

## 7. Survival analysis (su tutti i pazienti) — output = RISCHIO di morte CVD (prima la sopravvivenza)

> Tutta la sezione è impostata per **predire la sopravvivenza CVD a  anni** e poi per **predire/riportare la probabilità di MORTE CVD** (incidenza cumulata / failure function `F(t)=1−S(t)`), il contrario della funzione di sopravvivenza. Es. "rischio di morte CVD a 7 anni = X%". Il c-index **non cambia** col verso (ordina per rischio in entrambi i casi): si modifica solo ciò che si riporta e si disegna. Riporta anche le fasce di confidenza.


- **Incidenza cumulata di sopravvivenza** e **Incidenza cumulata di morte CVD** (lifelines): plotta `KM` e `1 − KM` (curva crescente) con il valore a 7 anni evidenziato. **Attenzione ai rischi competitivi**: `1 − KM` (che censura i decessi non-CVD) stima il rischio "netto" e lo **sovrastima**. Riporta anche l'estimatore corretto **Aalen-Johansen** (`AalenJohansenFitter`, evento di interesse = morte CVD, evento competitivo = morte non-CVD). Commenta lo scarto. Opzionale: modello **Fine-Gray** (subdistribution hazard) per le covariate sull'incidenza cumulata.
- **Stratificazione per rischio** (tertili del partial hazard del Cox su CV17): plotta le tre **curve di incidenza cumulata** crescenti e fai **log-rank** (atteso p ≈ 10⁻²¹¹). Rischio di sopravvivenza e poi morte CVD a 7a per gruppo (atteso): basso ≈ 0,02 / medio ≈ 0,08 / alto ≈ 0,30. → **GRAFICO**.
- **Cox** (lifelines) per interpretazione: **univariata** (HR per SD, p) e **multivariata** su tutte le combinazioni di feature. I tiroidei sono significativi da soli ma **perdono significatività aggiustati**. Esegui il **test di Schoenfeld** (proportional hazards) e segnala violazioni.
- **Predizione per-paziente**: dai modelli ricava il **rischio individuale di morte CVD a 7 anni** = `1 − S_i(7)` (per Cox/CoxNet/RSF/GBSurv usa la funzione di sopravvivenza predetta valutata a t=7).
- **c-index in 5-fold CV** (scikit-survival, `concordance_index_censored`) per `CoxPHSurvivalAnalysis`, `CoxnetSurvivalAnalysis`, `RandomSurvivalForest`, `GradientBoostingSurvivalAnalysis`, su tutte le combinazioni di features. Salva su `results_surv.csv`. *Nota risorse*: su 1 core RSF/GBSurv vanno alleggeriti (es. RSF n_estimators≈60, min_samples_leaf≈40, max_samples≈0.5; GBSurv n_estimators≈100, max_depth≈2).
- **GRAFICO**.

### 7-bis. Variante: survival SULLA STESSA coorte della classificazione

Oltre alla survival sul campione completo (tutti i pazienti), esegui una **variante ristretta alla coorte di classificazione** (strict e competing), per un confronto testa-a-testa coi modelli di classificazione *sugli stessi pazienti*:

- Tempo ed evento con **censura amministrativa a 7 anni**: `t = min(time_years, 7)`, `event = (event_cvd & time_years <= 7)`. Così l'analisi risponde esattamente alla domanda "morte CVD entro 7 anni", coerente col target binario. Output = **rischio di morte CVD a 7 anni** `1 − S_i(7)` per paziente (qui, con censura a 7a e — nella strict — senza eventi competitivi, `1 − KM` è già corretto).
- Calcola il **c-index in 5-fold CV** per Cox (almeno) e RSF, su tutti i set di features, per entrambe le coorti.
- **Caveat da riportare**: questa coorte rimuove i pazienti censurati prima dei 7 anni, quindi le stime **non** sono di rischio assoluto non distorto (c'è selezione sullo stato a 7 anni); la variante serve alla *comparabilità* con la classificazione, non alla stima epidemiologica. La survival sul campione completo (sez. 7) resta quella metodologicamente preferibile per il rischio assoluto.

## 8. SHAP analysis (richiesta esplicita)

Sul **modello tunato migliore** (e in parallelo su un modello ad albero, es. XGBoost/RandomForest, su cui SHAP è esatto e veloce), coorte **strict**, su tutti i set di features (così includi i tiroidei e ne misuri il contributo):

1. Allena il modello finale sul training set (con la pipeline; per SHAP applica l'explainer al classificatore su dati già scalati, senza il passo di sampling).
2. Usa `shap.TreeExplainer` per i modelli ad albero; per la LogisticRegression usa `shap.LinearExplainer` (o `KernelExplainer` su un campione di background ~100 istanze).
3. Produci:
   - **GRAFICO** — **beeswarm** (summary plot) di tutti i set di features: mostra direzione ed entità dell'effetto per ogni paziente.
   - **GRAFICO** — **bar plot** della SHAP importance media `mean(|SHAP|)`, evidenziando dove cadono le feature tiroidee.
   - **GRAFICO** — **dependence plot** per le 2-3 feature top (es. `Age`, `fe`) e per la migliore tiroidea (es. `fT4` o `Low_T3`), per leggere non-linearità e interazioni.
   - **Quantificazione**: somma di `mean(|SHAP|)` del blocco cardiaco vs blocco tiroideo; riporta la **percentuale di importanza totale attribuibile alla tiroide**. Confronta con la SHAP importance ottenuta su `CV17_CONT`.
4. **Lettura critica chiave**: Confronta la classifica SHAP con la **permutation importance** (sklearn) come check di robustezza.

## 9. Calibrazione (completa l'uso clinico)

Sul modello finale: `CalibratedClassifierCV` (isotonic e sigmoid), **Brier score** su test e **GRAFICO** calibration curve (10 bin) prima/dopo calibrazione.

## 10. Deliverable

- Notebook (più di uno) per mostrare pipeline e risultati
- Script modulari e ripristinabili: build dataset, screening, CV robusta, tuning, survival, SHAP, calibrazione.
- CSV dei risultati (`results_clf.csv`, `results_cv.csv`, `results_tune.csv`, `results_surv.csv`, `shap_importance.csv` e altri se opportuni).
- grafici.
- **Report** in markdown con: critica metodologica, cohort flow, EDA, risultati classificazione (tabella AUC/F1 per feature set), survival (c-index + Cox HR/p + KM), SHAP, calibrazione, e **conclusione**.


