
1. Prepare the data:
    - Download datasets [LEVIR](https://justchenhao.github.io/LEVIR/), [CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit), and [SYSU](https://github.com/liumency/SYSU-CD)

    Prepare datasets into the following structure and set their path in `train.py` and `test.py`
    ```
    ├─Train
        ├─A        ...jpg/png
        ├─B        ...jpg/png
        ├─label    ...jpg/png
        └─list     ...txt
    ├─Val
        ├─A
        ├─B
        ├─label
        └─list
    ├─Test
        ├─A
        ├─B
        ├─label
        └─list
    ```

2. Prerequisites for Python:
    - Creating a virtual environment in the terminal: `conda create -n mamba311 python=3.11`
    - Installing necessary packages: `pip install -r requirements.txt `

3. Train/Test
    - `sh ./train.sh`
    - `sh ./test.sh`






