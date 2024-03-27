from torcheval.metrics import MulticlassAUPRC
from torcheval.metrics.functional import multiclass_f1_score
from interpretable_ssl.pancras.dataset import PancrasDataset
import interpretable_ssl.utils as utils
import interpretable_ssl.pancras.train.vae_prototype_classifier as vae_prototype_classifier, interpretable_ssl.pancras.train.linear_classifier as pancras_linear_classifier
import torch


def evaluate(y, logits):
    y_pred = logits.argmax(dim=1)
    f1 = multiclass_f1_score(y, y_pred, num_classes=14, average="macro")
    auprc = MulticlassAUPRC(num_classes=14)
    auprc.update(logits, y_test)
    return f1, auprc.compute().cpu().numpy()


def main():
    device = utils.get_device()

    # load data
    pancras = PancrasDataset(device)
    train, test = pancras.get_train_test()

    # split to x, y
    x_test, y_test = test.dataset.x, test.dataset.y

    # load models
    # define model
    num_classes = 14
    num_prototypes, num_classes = 8, 14
    input_dim, hidden_dim, latent_dims = 4000, 64, 8
    protc = vae_prototype_classifier.ProtClassifier(
        num_prototypes=num_prototypes,
        num_classes=num_classes,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dims=latent_dims,
    )
    protc.load_state_dict(
        torch.load(vae_prototype_classifier.get_model_path(num_prototypes))[
            "model_state_dict"
        ]
    )
    protc.to(device)
    _, _, prot_logits = protc(x_test)
    f1, auprc = evaluate(y_test, prot_logits)
    print(f"prot f1:{f1}, auprc: {auprc}")

    linear = torch.nn.Sequential(torch.nn.Linear(8, 14, bias=False))
    linear.load_state_dict(
        torch.load(pancras_linear_classifier.get_model_path())["model_state_dict"]
    )
    linear.to(device)
    pancras.set_use_pca(True)
    train, test = pancras.get_train_test()

    # split to x, y
    x_test, y_test = test.dataset.x, test.dataset.y
    linear_logits = linear(x_test)
    f1, auprc = evaluate(y_test, linear_logits)
    print(f"linear f1:{f1}, auprc: {auprc}")
