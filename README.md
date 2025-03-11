# MLHD25
Bone Age Prediction from Hand X-Rays

https://pubs.rsna.org/doi/full/10.1148/radiol.2018180736

https://github.com/PooyaNasiri/bone-age-prediction?tab=readme-ov-file

Sarebbe bello generalizzarlo all'età ma non so se sia possibile

## First idea: CNN with 2 branches
- 2 branches
  - Pooled images 
    - Different branches inside for different pooling and kernels sizes
  - Heatmaps
    - Different branches inside

- **Pro**: Questa è una buona strategia perché sfrutta diverse rappresentazioni dei dati. Le immagini raggruppate (pooled images) possono catturare caratteristiche globali, mentre le mappe di calore possono evidenziare regioni specifiche rilevanti per la stima dell'età. L'utilizzo di diverse dimensioni di pooling e kernel permette di esplorare diverse scale di dettagli.
- **Contro**: La complessità del modello potrebbe aumentare, richiedendo più dati e tempo di addestramento. La creazione di mappe di calore accurate potrebbe essere impegnativa.

## Second idea: VAE con differenza sulle ricostruzioni
Mi piacerebbe fare un VAE con differenza sulle ricostruzioni 

- **Pro**: Un VAE (Variational Autoencoder) può apprendere una rappresentazione latente dei dati, che potrebbe essere utile per estrarre caratteristiche rilevanti per la stima dell'età. La differenza sulle ricostruzioni potrebbe aiutare a identificare anomalie o variazioni sottili nelle radiografie.
- **Contro**: L'addestramento di un VAE può essere complesso e richiedere una messa a punto accurata. La relazione tra la rappresentazione latente e l'età potrebbe non essere facilmente interpretabile.

## Third idea: Rete CNN con attenzione (Attention)
- **Descrizione**:
  - Utilizzare una rete CNN (Convolutional Neural Network) per estrarre caratteristiche dalle radiografie.
  - Aggiungere moduli di attenzione (attention) per permettere alla rete di concentrarsi sulle regioni più rilevanti delle immagini.
  - Utilizzare l'informazione sul sesso come input aggiuntivo alla rete.
  - Utilizzare una regressione per predire l'età.
- **Motivazione**:
  - Le CNN sono molto efficaci nell'elaborazione di immagini e possono apprendere automaticamente caratteristiche rilevanti.
  - I moduli di attenzione possono migliorare la capacità della rete di identificare le regioni delle radiografie più informative per la stima dell'età, come le epifisi delle ossa.
  - L'inclusione del sesso come input può migliorare l'accuratezza della previsione, poiché l'età scheletrica può variare tra uomini e donne.
  - L'utilizzo di un modello di regressione è appropriato per la previsione di un valore continuo come l'età.

## ChatGPT ideas:
Possibili Approcci Alternativi
1. Transfer Learning con Modelli Pre-addestrati
  
    Usare reti come ResNet, DenseNet o EfficientNet pre-addestrate su ImageNet può dare un ottimo punto di partenza. Queste reti sono molto efficaci nell'estrarre feature generiche e, con un fine-tuning sul tuo dataset, potresti ottenere buoni risultati anche con meno dati. Potresti aggiungere delle teste di regressione per stimare l'età, magari integrando il genere come input aggiuntivo.

2. Multi-task Learning

    Invece di trattare il genere come una side quest, potresti modellare il problema come una rete multi-task che, parallelamente alla regressione per l'età, predice anche il genere o altri indicatori di maturazione ossea. Questo approccio sfrutta la correlazione tra i compiti e, spesso, porta a una migliore generalizzazione.

3. Architetture Multi-scale

    Se l’idea della dual branch CNN (una per immagini pooled e una per heatmap) mira a catturare informazioni a scale diverse, potresti considerare l’uso di Feature Pyramid Networks (FPN). Queste architetture sono progettate per estrarre e combinare informazioni da diverse risoluzioni, offrendo un’integrazione più naturale tra dettagli globali e locali.

4. Capsule Networks

    Le Capsule Networks possono preservare la relazione spaziale tra le parti di un oggetto. In un contesto di radiografie, dove la disposizione delle ossa e delle epifisi è cruciale, questo potrebbe aiutare a modellare meglio le variazioni legate all’età. Naturalmente, il loro addestramento può risultare più complesso.

5. Self-supervised Learning

    Se il dataset non è enorme, potresti esplorare tecniche di self-supervised learning (come il contrastive learning) per pre-addestrare il network. Una volta apprese delle rappresentazioni robuste delle immagini radiografiche, potrai fine-tunare il modello sul compito di regressione per l’età.

6. Graph Neural Networks (GNN)

    Se hai la possibilità di segmentare le ossa o di estrarre landmark (punti di riferimento) dalle radiografie, un GNN potrebbe modellare le relazioni spaziali in maniera esplicita. Questo approccio è interessante perché potrebbe migliorare l’interpretabilità del modello, anche se richiede un pre-processing accurato per estrarre le strutture ossee rilevanti.

7. Modelli Ibridi Basati su GAN/VAE

    Unire tecniche generative (GAN o VAE) con una testa di regressione può avere il duplice vantaggio di generare dati sintetici per aumentare il dataset e di imparare una rappresentazione latente più ricca. Ad esempio, un Conditional VAE (CVAE) che condiziona la generazione anche sul genere potrebbe essere una strada da esplorare.