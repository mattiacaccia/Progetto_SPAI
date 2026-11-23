Progetto SPAI

Codice della tesi sulla tiny object detection e sul tracking delle teste degli spermatozoi nel dataset FertyLab, con SVIA usato come dataset aggiuntivo.

Vengono confrontati quattro modelli (YOLOv8, RT-DETR, DETR ViT-based, TOD-CNN) in due regimi: regime A (SVIA + FertyLab) e regime B (solo FertyLab). Lo split è fatto per campione: train SEM-0003, SEM-0007, SEM1; validation SEM-0004; test SEM-0002, SEM-0008. Sul modello migliore viene poi fatto il tracking con ByteTrack e l'analisi cinematica CASA.

Installazione

Serve uv quindi nel terminale: 

brew install uv

poi:

git clone https://github.com/mattiacaccia/Progetto_SPAI.git

cd Progetto_SPAI

uv sync


Dati

I dataset non sono nel repository. Vanno messi in data/:

data/DataSet_FertyLab/   (images/, yolo/labels/, csv/, manifest.json)

data/SVIA_DataSet/       (Subset-A/, Frames from original videos/)

Le altre cartelle in data/ vengono create dagli script.

Esecuzione

Preparazione dei dati (in quest'ordine nel temrinale):


uv run python scripts/parse_fertylab.py

uv run python scripts/parse_svia.py

uv run python src/datasets/slice_dataset.py

uv run python src/prepare_yolo_data.py

uv run python src/prepare_yolo_test.py

Training (i parametri sono in configs/train_config.yaml; per il regime B si usa regime_b):

uv run python src/train_ultralytics.py --model yolov8 --regime regime_a

uv run python src/train_ultralytics.py --model rtdetr --regime regime_a

uv run python src/train_detr_vit.py --regime regime_a

uv run python src/train_todcnn.py --regime regime_a

Test sul test set (--model può essere yolov8, rtdetr, detrvit, todcnn):

uv run python src/test.py --model yolov8 --regime regime_a

I risultati vengono salvati in runs/train/, runs/test/

Tracking e CASA (YOLOv8, Scenario A) sui video di data/spai_video/<CAMPIONE>/<CAMPIONE>_Vnn.avi con i relativi .camera.json. Lo split per campione è lo stesso della detection: train (SEM1, SEM-0003, SEM-0007) per stimare la velocità massima delle teste, validation (SEM-0004) per scegliere i parametri del tracker, test (tutti gli altri campioni, esclusi SEM-0012 e SEM-0013) per pseudo-etichette e CASA:

uv run python src/tracking.py --model yolov8 --regime regime_a --stage detect 

uv run python src/tracking.py --model yolov8 --regime regime_a --stage tune 

uv run python src/tracking.py --model yolov8 --regime regime_a --stage casa

I risultati vengono salvati in runs/tracking_video/.

DENO:
Interfaccia Streamlit con la object detection dei quattro modelli nei due scenari e il tracking con CASA sui video e tile di test:

uv run streamlit run app.py
