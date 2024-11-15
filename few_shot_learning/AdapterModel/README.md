### Using the Adapter Model

For the adapter model, we directly used the code from the original paper ([Compacter: Efficient Low-Rank Adaptation for Transformer Models](https://arxiv.org/abs/2106.04647)) available on this GitHub page:  
[https://github.com/rabeehk/compacter](https://github.com/rabeehk/compacter)

#### Steps to Run the Code on Your Data

1. **Clone the Repository**  
   Clone the original repository to your local machine:  
   ```bash
   git clone https://github.com/rabeehk/compacter.git
   
2. **Modify the Data Folder**  
   Replace the `tasks.py` file and add a new file, `crisis.py`, to the following directory in the repository:  
   ```bash
   compacter/seq2seq/data/
   ```
    
3. **Run the Script**  
   Navigate to the main folder of the repository:
   ```bash
   cd compacter/seq2seq/
   ```
   Then execute the adapter.sh script:
   ```
   ./adapter.sh
   ```

If you encounter any issues, please let us know by [sending an email](mailto:faezeghorbanpour96@gmail.com).