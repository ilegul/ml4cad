# ML4CAD - Report finale

## 1. Obiettivo e correzione metodologica

Obiettivo: predire la morte cardiovascolare entro 7 anni in pazienti con IHD, valutando se le variabili tiroidee aggiungono valore ai marker cardiovascolari.

La correzione metodologica centrale è stata applicata correttamente: il target binario non usa più il flag grezzo `CVD Death` su tutti i pazienti, perché quel flag ignora la censura a destra. Per la classificazione sono inclusi solo pazienti con stato a 7 anni osservabile:

- `y7=1`: morte CVD entro 7 anni.
- `y7=0`: event-free osservato ad almeno 7 anni.
- censurati prima di 7 anni: esclusi dalla coorte strict.
- morti non-CVD prima di 7 anni: escluse dalla strict, incluse come negativi nella competing.

La survival analysis usa invece tutti i pazienti con tempo informativo, perché la censura è gestita esplicitamente dal modello.

## 2. Correzioni data-quality

Le correzioni sono applicate nello step di preprocessing senza modificare i raw file. Il log è in `reports/data_quality_corrections.csv`.

- `Number=6850`: aveva `SCH=1` e `Hyperthyroid=1`. Scelta finale: `SCH=1`, `Hyperthyroid=0`, perché `TSH=4.76` è più coerente con ipotiroidismo subclinico che con ipertiroidismo.
- `Number=7286`: aveva `Data of death` presente ma `Total mortality=0`. Scelta finale: `Total mortality=1`, `CVD Death=0`, quindi morte non-CVD.
- `Number=6138`: CVD death a giorno 0. È mantenuto come evento CVD entro 7 anni. Per la survival viene incluso con tempo di analisi pari a 1 giorno, mantenendo anche il tempo originale tracciato.

Dopo la correzione, le categorie tiroidee sono mutualmente esclusive per tutti i 8065 pazienti.

## 3. Formati dati

I parquet restano il formato intermedio principale perché preservano tipi, date e colonne numeriche senza ambiguità e sono più efficienti. Per audit umano sono stati aggiunti anche CSV per tutte le coorti:

- `data/processed/cohort_full.parquet` e `.csv`
- `data/processed/cohort_strict.parquet` e `.csv`
- `data/processed/cohort_competing.parquet` e `.csv`
- `data/processed/cohort_survival.parquet` e `.csv`

## 4. Cohort flow

Numeri dopo correzioni:

- Full cohort: 8065 pazienti.
- CVD death entro 7 anni: 843.
- Event-free osservati ad almeno 7 anni: 3547.
- Censurati per classificazione: 3675.
- Alive con follow-up <7 anni: 2628.
- Morti non-CVD <7 anni: 1047.
- Strict cohort: 4390, positivi 843, prevalenza 19,2%.
- Competing cohort: 5437, positivi 843, prevalenza 15,5%.
- Survival cohort: 8063, eventi CVD 1008.

Figura: `reports/figures/cohort_flow.png`.

## 5. EDA

Nella coorte strict non ci sono missing nelle 26 feature principali. La creatinina resta incompleta ed è esclusa dai feature set principali.

Associazioni univariate principali sul target strict:

- `PostIsch_DCM`: OR circa 5,98.
- `Previous_CABG`: OR circa 2,57.
- `Acute_MI`: OR circa 2,51.
- `Diabetes`: OR circa 2,27.
- `AFib`: OR circa 2,25.
- Tiroide: `SCH` OR circa 1,91, `Low_T3` OR circa 1,70.
- Continue: `Age` e `fe` hanno gli effetti più forti tra le continue.

Nota: odds ratio e rank-biserial non vanno letti come una graduatoria unica perfettamente confrontabile; sono scale diverse.

Clustering pazienti: KMeans su feature standardizzate ha silhouette massima circa 0,124. Quindi non emerge una struttura di cluster forte; la PCA mostra sovrapposizione sostanziale tra classi.

Figure:

- `reports/figures/violins_continuous.png`
- `reports/figures/dendrogram_features.png`
- `reports/figures/heatmap_spearman.png`
- `reports/figures/pca_clustering.png`

## 6. Classificazione

Metrica principale: F1-macro. AUC e PR-AUC sono considerate metriche secondarie.

Lo screening single split è completo: 640/640 combinazioni, senza duplicati. La robust CV è completa: 64/64 combinazioni, senza duplicati.

### Robust CV 5-fold

Strict cohort, migliori risultati per F1-macro:

- `CV17_RATIO_ONLY` + LogisticRegression: F1-macro 0,738 ± 0,022; AUC 0,840 ± 0,016.
- `CV17` + LogisticRegression: F1-macro 0,738 ± 0,031; AUC 0,840 ± 0,016.
- `CV17_CONT` + LogisticRegression: F1-macro 0,733 ± 0,027; AUC 0,840 ± 0,016.
- `CV17_THY26` + LogisticRegression: F1-macro 0,733 ± 0,022; AUC 0,838 ± 0,016.

Competing cohort, migliori risultati per F1-macro:

- `CV17_RATIO` + LogisticRegression: F1-macro circa 0,659.
- `CV17` + LogisticRegression: F1-macro circa 0,658.
- `CV17_RATIO_ONLY` + LogisticRegression: F1-macro circa 0,657.
- `CV17_THY26` + LogisticRegression: F1-macro circa 0,657.

Interpretazione: le feature tiroidee non producono un miglioramento predittivo robusto in classificazione. Le curve dei feature set sono quasi sovrapposte; i piccoli delta stanno entro la variabilità tra fold.

### Tuning

Il miglior tuning strict per F1-macro è:

- `CV17_THY26` + HistGradientBoosting + SVMSMOTE: CV-F1-macro 0,7395.

Questo valore è solo marginalmente superiore alla Logistic Regression in robust CV e va interpretato con cautela. RandomForest mostra train AUC molto elevati in tuning, segnale di possibile overfitting; per confronto scientifico il riferimento resta la CV robusta.

Figure:

- `reports/figures/f1_by_feature_set.png`
- `reports/figures/auc_by_feature_set.png`

## 7. Survival analysis

Sul campione completo:

- Kaplan-Meier survival a 7 anni: `S(7)=0,8744`.
- Rischio CVD a 7 anni da `1-KM`: `F(7)=0,1256`.
- Aalen-Johansen competing-risk CVD a 7 anni: `0,1172`.
- Sovrastima di `1-KM` rispetto ad Aalen-Johansen: 0,0083 assoluti, circa 7,1% relativo.

Quindi per rischio assoluto con competing risks va preferita Aalen-Johansen; `1-KM` è utile come rischio netto ma sovrastima l'incidenza reale.

### Stratificazione per rischio

Cox su `CV17`, terzili del partial hazard:

- Low risk: n=2688, rischio CVD a 7 anni circa 2,27%.
- Medium risk: n=2687, rischio circa 8,26%.
- High risk: n=2688, rischio circa 29,59%.
- Log-rank Low vs High: p circa `1,22e-167`.
- Log-rank globale: p circa `1,18e-210`.

Figura: `reports/figures/risk_stratification.png`.

### Cox multivariata

La Cox multivariata ora usa `Euthyroid` come categoria di riferimento, evitando la collinearità precedente. Risultati principali:

- `Age`: HR circa 1,99 per SD.
- `fe`: HR circa 0,56 per SD.
- `Diabetes`: HR circa 1,46.
- `Dyslipidemia`: HR circa 0,70.
- `AFib`: HR circa 1,27.
- `SCH`: HR circa 1,54 vs euthyroid.
- `SCT`: HR circa 1,48 vs euthyroid.
- `Low_T3`: HR circa 1,52 vs euthyroid.
- `TSH` e `fT3` non sono significativi dopo aggiustamento; `fT4` resta debolmente significativo.

Il test di Schoenfeld segnala violazioni dell'assunzione di proportional hazards per `Previous_CABG`, `Acute_MI`, `Dyslipidemia`, `SCT`, `Low_T3`. Questo non invalida l'intera analisi, ma limita l'interpretazione causale/temporale dei relativi HR; un'estensione naturale è una Cox stratificata o con termini tempo-dipendenti.

### Survival CV

Full survival cohort, migliori c-index:

- `CV17_BIN` + CoxNet: 0,7939 ± 0,0123.
- `CV17_BIN` + CoxPH: 0,7938 ± 0,0122.
- `CV17_CAT` + CoxNet: 0,7934 ± 0,0123.
- `CV17_THY26` + CoxPH/CoxNet: circa 0,7929.

La precedente anomalia CoxNet=0,50 è risolta.

Survival su coorti di classificazione:

- Strict, migliore: `CV17_BIN` + CoxPH, c-index 0,8123 ± 0,0157.
- Competing, migliore: `CV17_BIN` + CoxPH, c-index 0,7846 ± 0,0136.

Questa variante serve solo per confronto con la classificazione: rimuove pazienti censurati prima dei 7 anni e non va usata come stima epidemiologica assoluta.

Figure:

- `reports/figures/km_cumulative_incidence.png`
- `reports/figures/aalen_johansen.png`
- `reports/figures/survival_cindex.png`

## 8. SHAP e permutation importance

SHAP con XGBoost 3.2 non era compatibile con il formato `base_score` serializzato. È stato quindi usato RandomForest con `TreeExplainer`, scelta robusta e coerente con un modello tree-based.

Percentuale di importanza SHAP attribuita al blocco tiroideo:

- `CV17_BIN`: 6,0%.
- `CV17_ORD`: 1,6%.
- `CV17_CAT`: 4,5%.
- `CV17_CONT`: 13,3%.
- `CV17_THY26`: 13,9%.
- `CV17_RATIO`: 15,7%.
- `CV17_RATIO_ONLY`: 16,5%.

Controllo SHAP vs permutation importance:

- `CV17_THY26`: correlazione circa 0,903.
- `CV17_RATIO`: correlazione circa 0,940.

Interpretazione: RandomForest assegna un contributo non nullo alle variabili tiroidee, specialmente quando si includono TSH/fT3/fT4/ratio. Tuttavia questo contributo non si traduce in un miglioramento robusto di F1-macro in classificazione. Quindi le feature tiroidee sono informative/associate, ma il loro valore incrementale predittivo rispetto a `CV17` è debole.

Output:

- `data/processed/shap_importance.csv`
- `data/processed/permutation_importance.csv`
- `reports/figures/shap_*.png`

## 9. Calibrazione

Modello finale selezionato per F1-macro tuned strict:

- HistGradientBoosting + `CV17_THY26` + SVMSMOTE.

Risultati su test:

- Base: Brier 0,1166; AUC 0,8240; F1-macro 0,7436.
- Isotonic: Brier 0,1081; AUC 0,8344; F1-macro 0,7080.
- Sigmoid: Brier 0,1079; AUC 0,8383; F1-macro 0,6999.

Interpretazione: la calibrazione migliora chiaramente la qualità probabilistica (Brier più basso), ma peggiora la F1-macro alla soglia predefinita. Per probabilità cliniche va preferito il modello calibrato; per decisione binaria va scelta e validata una soglia esplicita.

Figura: `reports/figures/calibration_curve.png`.

## 10. Conclusione

Il lavoro è ora metodologicamente molto più solido: censura gestita, competing risks esplicitati, pipeline rigenerata con `.venv`, data-quality tracciata, SHAP e calibrazione completate.

Conclusione scientifica principale:

1. Le feature cardiovascolari `CV17` già catturano la maggior parte del segnale predittivo.
2. Le variabili tiroidee sono associate all'outcome, e in Cox alcune restano significative dopo aggiustamento.
3. In classificazione, però, il valore incrementale delle variabili tiroidee è piccolo e non robusto: F1-macro e AUC restano quasi sovrapposti tra `CV17` e i set arricchiti.
4. Il rischio assoluto CVD a 7 anni va riportato con Aalen-Johansen quando si considerano morti non-CVD come competing risk.
5. Per uso clinico, la calibrazione va usata per probabilità; la F1-macro resta la metrica primaria per scelta del classificatore binario.

Limiti residui:

- Alcune covariate violano proportional hazards.
- Il tuning non è nested rispetto alla stima finale delle performance.
- Le soglie decisionali andrebbero selezionate con validazione interna dedicata o nested CV.
- Le stime su coorti strict/competing sono utili per comparabilità, non per rischio assoluto non distorto.
