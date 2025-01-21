from interpretable_ssl.models.linear import *

class Classifier:
    def __init__(self, trainer, transform_x=None, single_cell=False, cells=None, genes=None):
        
        if transform_x is None:
            self.transform_x = trainer.encode_adata
        else:
            self.transform_x = transform_x
        
        self.trainer = trainer
        self.clf = None
        self.classes_ = None
        self.single_cell = single_cell
        self.cells = cells
        self.genes = genes
        self.proto_cells = None
        self.proto_train = False
        self.trainer.model = self.trainer.load_model()
        
    def get_x_y(self, adata):
        if self.cells is not None:
            cell_idx = self.get_cell_filter_idx(adata)
            adata = adata[cell_idx]
        x = self.transform_x(adata)
        y = adata.obs["cell_type"]
        return x, y
    
    def prepare_data(self):
        if self.proto_train:
            print("Preparing prototype data")
            x_train, y_train = self.prepare_proto_data(self.trainer.ref.adata)
            x_test, y_test = self.prepare_proto_data(self.trainer.query.adata)
        elif self.single_cell:
            print("Preparing single cell data")
            x_train, y_train = self.prepare_single_cell_data(self.trainer.ref.adata)
            x_test, y_test = self.prepare_single_cell_data(self.trainer.query.adata)
            
        else:
            x_train, y_train = self.get_x_y(self.trainer.ref.adata)
            x_test, y_test = self.get_x_y(self.trainer.query.adata)
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        return x_train, y_train, x_test, y_test
    
    def train_classifier(self):
        x_train, y_train, x_test, y_test = self.prepare_data()
        self.clf = LinearClassifier(X_train=x_train, X_test=x_test, y_train=y_train, y_test=y_test)
        self.classes_ = self.clf.classes_
        self.clf.train()
        return self.clf.evaluate()
    
    def get_cell_filter_idx(self, adata):
        return adata.obs.cell_type.isin(self.cells) 
    
    def get_gene_indices(self, adata):
        
        return [np.where(adata.var_names == gene)[0][0] for gene in self.genes if gene in adata.var_names]
    
    def prepare_single_cell_data(self, adata):
        cell_idx = self.get_cell_filter_idx(adata)
        gene_indices = self.get_gene_indices(adata)
        x = adata.X[cell_idx, :][:, gene_indices].toarray() 
        y = adata[cell_idx].obs["cell_type"]
        return x, y

    def get_proto_cells(self):
        return self.trainer.model.decode_proto(self.trainer.recon_loss, True)

    def prepare_proto_data(self, adata):
        cell_idx = self.get_cell_filter_idx(adata)
        proto_idx = self.trainer.encode_adata(adata[cell_idx], self.trainer.model, True, True)
        gene_indices = self.get_gene_indices(adata)
        if self.proto_cells is None:
            self.proto_cells = self.get_proto_cells()
        return self.proto_cells[proto_idx, :][:, gene_indices], adata[cell_idx].obs["cell_type"]