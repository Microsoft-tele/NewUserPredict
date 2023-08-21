import yaml
with open("./config.yaml", "r") as f:
    config_file = yaml.safe_load(f)
class Config:
    def __init__(self,i):
        self.train_unknown_csv_filepath= config_file["train_unknown_csv_filepath"]
        self.train_unknown_save_path= config_file["train_unknown_csv_filepath"]
        self.train_csv= config_file["train_csv"]
        self.train_unknown_csv= config_file["train_unknown_csv"]
        self.train_known_csv= config_file["train_known_csv"]
        self.analyse_read_csv= config_file["analyse_read_csv"]
        self.processed_fil_1= config_file["processed_fil_1"]
        self.processed_fil_2= config_file["processed_fil_2"]
        self.processed_fil_3= config_file["processed_fil_3"]
        self.processed_fil_4= config_file["processed_fil_4"]
        self.log_analyse= config_file["log_analyse"]
        self.output_key1 = config_file["output_key1"]
        self.output_key2 = config_file["output_key2"]
        self.output_key3 = config_file["output_key3"]
        self.output_key4 = config_file["output_key4"]
        self.output_key5 = config_file["output_key5"]
        self.output_key6 = config_file["output_key6"]
        self.output_key7 = config_file["output_key7"]
        self.output_key8 = config_file["output_key8"]
        self.output_key9 = config_file["output_key9"]
        self.input_size= 10
        self.lr= 0.0001
        self.num_epochs= 100
        self.batch_size= 512