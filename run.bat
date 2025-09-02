REM '1 small version'
python main.py ^
    --work fit ^
    --cfg ./cfgs/mff_s.yaml ^
    --data-root ../../datasets/segment ^
    --batch-size 8 ^
    --grad_freq 1 ^

REM '2 tiny version'
python main.py ^
    --work fit ^
    --cfg ./cfgs/mff_t.yaml ^
    --data-root ../../datasets/segment ^
    --batch-size 8 ^
    --grad_freq 1 ^

REM '3 base version'
python main.py ^
    --work fit ^
    --cfg ./cfgs/mff_b.yaml ^
    --data-root ../../datasets/segment ^
    --batch-size 4 ^
    --grad_freq 2 ^
    
REM '4 tiny version, disable all'
python main.py ^
    --work fit ^
    --cfg ./cfgs/mff_t.yaml ^
    --mlla-disable ^
    --fft-disable ^
    --gate-disable ^
    --data-root ../../datasets/segment ^
    --batch-size 8 ^
    --grad_freq 1 ^

REM '5 tiny version, enable mlla'
python main.py ^
    --work fit ^
    --cfg ./cfgs/mff_t.yaml ^
    --fft-disable ^
    --gate-disable ^
    --data-root ../../datasets/segment ^
    --batch-size 8 ^
    --grad_freq 1 ^
    
REM '6 tiny version, enable mlla, fft'
python main.py ^
    --work fit ^
    --cfg ./cfgs/mff_t.yaml ^
    --gate-disable ^
    --data-root ../../datasets/segment ^
    --batch-size 8 ^
    --grad_freq 1 ^