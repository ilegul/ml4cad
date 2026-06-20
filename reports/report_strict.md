# ML4CAD - Report coorte strict

Data report: 18 giugno 2026

## 1. Scopo e perimetro

Questo report valuta esclusivamente la coorte `strict`, costruita per la domanda binaria:

> morte cardiovascolare entro 7 anni, usando solo pazienti con stato a 7 anni osservabile.

La domanda scientifica e' se uno dei feature set con variabili tiroidee migliori la predizione rispetto al set cardiovascolare di base `CV17`, in classificazione e nella survival analysis ristretta alla stessa coorte.

Sono volutamente esclusi dal corpo del report i risultati della coorte `competing` e della survival sul campione completo, salvo quando servono per segnalare una mancanza metodologica.

## 2. Audit metodologico del flusso

### 2.1 Cosa e' corretto

La costruzione del target strict e' corretta. Nel preprocessing:

- `y7=1` se `event_cvd==1` e `time_days <= 7 anni`;
- `y7=0` se il paziente e' osservato event-free almeno fino a 7 anni;
- i pazienti censurati prima di 7 anni e le morti non-CVD prima di 7 anni sono esclusi dalla strict.

Verifica numerica:

| Controllo | Valore |
|---|---:|
| Full cohort | 8065 |
| CVD death <=7 anni | 843 |
| Event-free >=7 anni | 3547 |
| Coorte strict attesa | 4390 |
| Coorte strict prodotta | 4390 |
| Membership strict corretta | si |
| Positivi strict | 843 |
| Negativi strict | 3547 |
| Prevalenza evento strict | 19,2% |

Le categorie tiroidee sono mutuamente esclusive dopo le correzioni data-quality. Nella strict:

| Stato tiroideo | N |
|---|---:|
| Euthyroid | 2902 |
| Low_T3 | 910 |
| SCH | 261 |
| SCT | 155 |
| Hyperthyroid | 87 |
| Hypothyroid | 75 |

Non ci sono missing nelle 26 feature principali della coorte strict.

La classificazione evita leakage evidente: lo scaling e il sampling sono dentro una `imblearn.Pipeline`, quindi il sampler viene fit solo sul training split/fold. Anche la copertura sperimentale e' completa:

| Blocco | Atteso strict | Prodotto strict | Duplicati |
|---|---:|---:|---:|
| Screening classificazione | 320 | 320 | 0 |
| Robust CV classificazione | 32 | 32 | 0 |
| Tuning | 12 | 12 | 0 |
| Survival strict CV | 16 | 16 | 0 |

### 2.2 Errori o mancanze da correggere

1. La calibrazione non riapplica gli iperparametri migliori del tuning.

   `calibration.py` seleziona il miglior `feature_set/model/sampler` da `results_tune.csv`, ma poi ricostruisce il modello con `_build_pipeline(model_name, sampler_name)`, ignorando `best_params`. Quindi i risultati di calibrazione sono riferiti alla famiglia/modello selezionata, non al modello tunato effettivo.

2. La comparison SHAP vs permutation importance e' etichettata come Spearman ma calcola Pearson.

   In `shap_analysis.py` viene chiamato `.corr()` senza `method="spearman"`, quindi il valore stampato come "Spearman corr" e' Pearson. Ricalcolando Spearman via rank, per esempio:

   | Feature set | Pearson | Spearman corretto |
   |---|---:|---:|
   | CV17_THY26 | 0,903 | 0,362 |
   | CV17_RATIO | 0,940 | 0,825 |

   L'accordo SHAP/permutation resta buono per `CV17_RATIO`, ma e' molto meno convincente per `CV17_THY26`.

3. Il tuning non e' nested rispetto alla stima finale delle performance.

   Il campo `cv_f1_macro` e' il valore piu' utile del tuning, ma `train_f1_opt_thr` e `train_auc` sono calcolati sullo stesso dataset usato per fit/refit e sono ottimistici. Non vanno usati come performance generalizzabile.

4. L'ottimizzazione della soglia nella robust CV usa predizioni in-sample sul training fold.

   E' accettabile come scelta pragmatica, ma una stima piu' rigorosa dovrebbe scegliere la soglia con una validazione interna o nested CV.

5. Non esiste un output Cox interpretativo dedicato alla strict.

   I file `cox_univariate.csv` e `cox_multivariate.csv` sono prodotti dalla survival sul campione completo. Per un report strict-only non vanno usati come evidenza principale. La strict ha solo c-index Cox/RSF con censura amministrativa a 7 anni.

6. SMOTE/SVMSMOTE su feature binarie/categoriche crea valori sintetici non fisiologici.

   Il sampler e' correttamente dentro la CV, quindi non c'e' leakage, ma la natura delle feature suggerisce di confrontare anche `class_weight`, undersampling, oppure `SMOTENC` con maschera delle categoriche.

## 3. Definizione della coorte strict

La strict risponde alla domanda di classificazione in modo pulito: evento CVD entro 7 anni vs event-free osservato almeno 7 anni. Sono rimossi:

- vivi con follow-up inferiore a 7 anni;
- deceduti per causa non-CVD prima dei 7 anni.

Per la variante survival strict e' stata applicata censura amministrativa a 7 anni:

| Quantita' | Valore |
|---|---:|
| N strict | 4390 |
| Eventi CVD entro 7 anni | 843 |
| Censurati amministrativamente a 7 anni | 3547 |
| Rischio grezzo nella coorte selezionata | 19,2% |

Nota importante: questa survival strict serve per confrontare modelli sugli stessi pazienti della classificazione. Non stima il rischio assoluto non distorto nella popolazione, perche' la coorte e' selezionata sul fatto di avere stato a 7 anni osservabile.

## 4. Feature set confrontati

Il riferimento e' `CV17`, composto da 17 variabili cardiovascolari. I set con tiroide aggiungono le informazioni in forme diverse:

| Feature set | Informazione tiroidea aggiunta |
|---|---|
| CV17_THY26 | 9 variabili tiroidee raw |
| CV17_BIN | `Thyroid_abnormal` |
| CV17_ORD | `thyroid_ord` |
| CV17_CONT | `TSH`, `fT3`, `fT4` |
| CV17_CAT | categorie tiroidee separate |
| CV17_RATIO | raw9 + `fT3_fT4_ratio` |
| CV17_RATIO_ONLY | `TSH`, `fT3`, `fT4`, ratio |

## 5. EDA strict

Le associazioni univariate confermano che il segnale principale e' cardiovascolare. Le variabili tiroidee sono associate, ma con effetto inferiore ai marker cardiaci principali.

Top associazioni univariate con `y7`:

| Feature | Tipo | Effetto | p-value |
|---|---|---:|---:|
| PostIsch_DCM | binaria | OR 5,98 | 3,74e-74 |
| Previous_CABG | binaria | OR 2,57 | 8,17e-17 |
| Acute_MI | binaria | OR 2,51 | 2,47e-14 |
| Diabetes | binaria | OR 2,27 | 1,05e-22 |
| AFib | binaria | OR 2,25 | 1,05e-20 |
| SCH | tiroide | OR 1,91 | 4,25e-06 |
| Previous_MI | binaria | OR 1,90 | 8,53e-16 |
| Low_T3 | tiroide | OR 1,70 | 1,68e-09 |
| Hypothyroid | tiroide | OR 1,65 | 0,071 |
| Hyperthyroid | tiroide | OR 1,62 | 0,062 |

Questa e' evidenza di associazione, non di valore incrementale predittivo. Il confronto predittivo va letto dalla CV.

Il clustering KMeans su feature standardizzate non mostra struttura forte: silhouette massima circa 0,124. La PCA mostra sovrapposizione importante tra classi, quindi non emerge un sottogruppo naturale netto che separi eventi e non-eventi.

## 6. Classificazione strict

### 6.1 Screening single split

Lo screening 70/30 e' utile per esplorare molte combinazioni, ma non deve guidare la conclusione finale. I migliori risultati single split sono:

| Rank | Feature set | Modello | Sampler | F1-macro | ROC-AUC | PR-AUC |
|---:|---|---|---|---:|---:|---:|
| 1 | CV17_CONT | RandomForest | SVMSMOTE | 0,750 | 0,835 | 0,640 |
| 2 | CV17_RATIO | HistGradientBoosting | BorderlineSMOTE | 0,747 | 0,823 | 0,616 |
| 3 | CV17_CONT | RandomForest | SMOTE | 0,746 | 0,830 | 0,626 |
| 4 | CV17_RATIO_ONLY | RandomForest | SVMSMOTE | 0,744 | 0,831 | 0,628 |
| 5 | CV17_RATIO | HistGradientBoosting | SVMSMOTE | 0,744 | 0,827 | 0,619 |
| 6 | CV17_THY26 | HistGradientBoosting | SVMSMOTE | 0,744 | 0,824 | 0,607 |

Lo screening suggerisce che alcuni set con tiroide possono arrivare in alto in specifiche combinazioni modello/sampler, ma questo vantaggio non si conferma in modo robusto nella CV.

### 6.2 Robust CV 5-fold

Metrica primaria: F1-macro con soglia ottimizzata. Tutti i migliori risultati per feature set sono LogisticRegression + SMOTE.

| Feature set | Modello migliore | F1-macro media | F1-macro sd | ROC-AUC media | ROC-AUC sd |
|---|---|---:|---:|---:|---:|
| CV17_RATIO_ONLY | LogisticRegression | 0,738 | 0,022 | 0,840 | 0,016 |
| CV17 | LogisticRegression | 0,738 | 0,030 | 0,840 | 0,016 |
| CV17_CONT | LogisticRegression | 0,733 | 0,027 | 0,840 | 0,016 |
| CV17_BIN | LogisticRegression | 0,733 | 0,026 | 0,839 | 0,016 |
| CV17_RATIO | LogisticRegression | 0,733 | 0,019 | 0,838 | 0,017 |
| CV17_THY26 | LogisticRegression | 0,732 | 0,022 | 0,838 | 0,016 |
| CV17_CAT | LogisticRegression | 0,732 | 0,031 | 0,839 | 0,016 |
| CV17_ORD | LogisticRegression | 0,732 | 0,023 | 0,839 | 0,016 |

Il confronto piu' importante e' contro `CV17`:

| Set | F1-macro | Delta F1 vs CV17 | ROC-AUC | Delta AUC vs CV17 |
|---|---:|---:|---:|---:|
| CV17 | 0,7378 | riferimento | 0,8397 | riferimento |
| CV17_RATIO_ONLY | 0,7381 | +0,0003 | 0,8397 | -0,0000 |
| CV17_CONT | 0,7334 | -0,0044 | 0,8396 | -0,0001 |
| CV17_BIN | 0,7328 | -0,0050 | 0,8394 | -0,0003 |
| CV17_RATIO | 0,7326 | -0,0052 | 0,8377 | -0,0019 |
| CV17_THY26 | 0,7325 | -0,0053 | 0,8377 | -0,0020 |

Conclusione classificazione strict: non c'e' evidenza robusta che i set tiroidei migliorino il modello. Il massimo incremento F1 rispetto a `CV17` nella migliore famiglia di modello e' praticamente nullo per la LogisticRegression (`+0,0003`) e molto piu' piccolo della deviazione standard tra fold.

Alcuni modelli non lineari beneficiano localmente delle feature tiroidee rispetto alla loro versione `CV17`: per RandomForest, `CV17_THY26` migliora F1 di circa `+0,014` e AUC di circa `+0,010`. Tuttavia RandomForest resta sotto la LogisticRegression `CV17`, quindi questo non cambia la conclusione generale.

### 6.3 Tuning

Il miglior tuning strict e':

| Feature set | Modello | Sampler | CV F1-macro |
|---|---|---|---:|
| CV17_THY26 | HistGradientBoosting | SVMSMOTE | 0,7395 |
| CV17_THY26 | XGBoost | SVMSMOTE | 0,7341 |
| CV17_CONT | RandomForest | SVMSMOTE | 0,7339 |
| CV17_CONT | XGBoost | SVMSMOTE | 0,7320 |
| CV17 | XGBoost | SVMSMOTE | 0,7311 |

Il valore migliore (`0,7395`) e' solo `+0,0017` sopra la robust CV di `CV17` con LogisticRegression (`0,7378`) e non e' nested. Lo considero un segnale esplorativo, non una dimostrazione di miglioramento.

## 7. Survival analysis strict

La survival strict usa gli stessi pazienti della classificazione e applica censura amministrativa a 7 anni:

`t = min(time_years, 7)`, `event = event_cvd entro 7 anni`.

Sono stati valutati CoxPH e Random Survival Forest su tutti gli 8 feature set.

| Feature set | Modello | c-index medio | sd |
|---|---|---:|---:|
| CV17_BIN | CoxPH | 0,8123 | 0,0157 |
| CV17 | CoxPH | 0,8117 | 0,0175 |
| CV17_CAT | CoxPH | 0,8117 | 0,0158 |
| CV17_ORD | CoxPH | 0,8116 | 0,0175 |
| CV17_RATIO_ONLY | CoxPH | 0,8115 | 0,0168 |
| CV17_CONT | CoxPH | 0,8114 | 0,0167 |
| CV17_RATIO | CoxPH | 0,8105 | 0,0161 |
| CV17_THY26 | CoxPH | 0,8105 | 0,0156 |
| CV17_CONT | RSF | 0,8087 | 0,0185 |
| CV17_THY26 | RSF | 0,8080 | 0,0191 |
| CV17_BIN | RSF | 0,8074 | 0,0168 |
| CV17_RATIO | RSF | 0,8074 | 0,0169 |
| CV17_RATIO_ONLY | RSF | 0,8072 | 0,0183 |
| CV17_CAT | RSF | 0,8067 | 0,0150 |
| CV17 | RSF | 0,8064 | 0,0176 |
| CV17_ORD | RSF | 0,8060 | 0,0188 |

Delta rispetto a `CV17` nello stesso modello:

| Modello | Miglior set tiroideo | Delta c-index vs CV17 |
|---|---|---:|
| CoxPH | CV17_BIN | +0,0006 |
| RSF | CV17_CONT | +0,0023 |

Conclusione survival strict: nessun set tiroideo migliora in modo materialmente rilevante il c-index. I delta sono nell'ordine di `0,001-0,002`, molto inferiori alla variabilita' tra fold (`sd` circa `0,016-0,019`).

Mancanza da colmare se serve interpretazione clinica strict: produrre Cox multivariata e test di Schoenfeld sulla strict amministrativamente censurata, senza riusare i file Cox full-sample.

## 8. SHAP e permutation importance strict

La SHAP analysis e' stata eseguita sulla strict con RandomForest e TreeExplainer per tutti i feature set. Non e' il modello tunato migliore, quindi va letta come analisi di sensibilita' interpretativa, non come spiegazione unica del modello finale.

Quota di importanza SHAP attribuita al blocco tiroideo:

| Feature set | Importanza tiroide |
|---|---:|
| CV17_BIN | 6,0% |
| CV17_ORD | 1,6% |
| CV17_CAT | 4,5% |
| CV17_CONT | 13,3% |
| CV17_THY26 | 13,9% |
| CV17_RATIO | 15,7% |
| CV17_RATIO_ONLY | 16,5% |

Le feature tiroidee piu' importanti sono soprattutto `fT4`, `TSH`, `fT3` e `fT3_fT4_ratio`. Le top feature globali restano pero' cardiovascolari o cliniche generali: `Age`, `fe`, `Diabetes`, `Dyslipidemia`, `Vessels`, `PostIsch_DCM`.

Interpretazione: la tiroide porta segnale nel modello RandomForest, specialmente quando e' rappresentata con valori continui e ratio. Questo segnale non si traduce pero' in miglioramento robusto di F1-macro o c-index rispetto a `CV17`.

La comparison con permutation importance richiede correzione: il codice stampa "Spearman" ma calcola Pearson. Con Spearman corretto l'accordo e' variabile:

| Feature set | Spearman corretto SHAP vs permutation |
|---|---:|
| CV17_THY26 | 0,362 |
| CV17_CONT | 0,759 |
| CV17_RATIO | 0,825 |
| CV17_RATIO_ONLY | 0,566 |

Quindi l'importanza tiroidea e' plausibile per i set continui/ratio, ma non abbastanza stabile da sostenere da sola un claim forte.

## 9. Calibrazione strict

Il modello selezionato dal tuning strict e' `HistGradientBoosting + CV17_THY26 + SVMSMOTE`. Tuttavia, per il bug indicato sopra, la calibrazione usa la configurazione default della pipeline, non gli iperparametri migliori salvati in `best_params`.

Risultati sul test split strict:

| Metodo | Brier | ROC-AUC | F1-macro |
|---|---:|---:|---:|
| Base | 0,1166 | 0,8240 | 0,7436 |
| Isotonic | 0,1081 | 0,8344 | 0,7080 |
| Sigmoid | 0,1079 | 0,8383 | 0,6999 |

La calibrazione migliora il Brier score e l'AUC, ma peggiora la F1 alla soglia predefinita. Questo e' coerente: calibrare probabilita' e ottimizzare una decisione binaria sono obiettivi diversi. Dopo aver corretto l'uso dei `best_params`, la soglia decisionale andrebbe scelta di nuovo su validation interna.

## 10. Risposta alla domanda scientifica

Sulla coorte strict, le feature tiroidee sono associate all'outcome e contribuiscono alle spiegazioni SHAP, ma non migliorano in modo robusto la predizione rispetto a `CV17`.

Classificazione:

- `CV17` con LogisticRegression + SMOTE: F1-macro `0,7378`, AUC `0,8397`.
- Miglior set tiroideo robust CV: `CV17_RATIO_ONLY`, F1-macro `0,7381`, AUC `0,8397`.
- Delta pratico: nullo.
- Il miglior tuning con tiroide arriva a `0,7395`, ma il vantaggio e' minimo e non nested.

Survival strict:

- `CV17` con CoxPH: c-index `0,8117`.
- Miglior set tiroideo CoxPH: `CV17_BIN`, c-index `0,8123`, delta `+0,0006`.
- Miglior set tiroideo RSF: `CV17_CONT`, delta `+0,0023`.
- Delta pratico: non rilevante.

Conclusione senior:

> Nella coorte strict non c'e' evidenza sufficiente per affermare che l'aggiunta delle variabili tiroidee migliori la classificazione o la survival prediction a 7 anni rispetto alle sole feature cardiovascolari `CV17`. Le variabili tiroidee sembrano informative e clinicamente interessanti, ma il loro valore incrementale predittivo e' debole, instabile e inferiore alla variabilita' della validazione.

## 11. Raccomandazioni operative

Prima di usare i risultati come finali:

1. correggere `calibration.py` per applicare i `best_params` del tuning;
2. correggere `shap_analysis.py` usando `method="spearman"` o cambiando etichetta in Pearson;
3. aggiungere Cox multivariata strict con censura amministrativa a 7 anni e test di Schoenfeld strict;
4. ripetere tuning e soglia con nested CV o validation interna dedicata;
5. confrontare SMOTE/SVMSMOTE con `class_weight`, undersampling e/o `SMOTENC`;
6. riportare come claim principale il risultato robust CV, non lo screening single split.

