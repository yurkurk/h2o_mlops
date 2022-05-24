import h2o
from h2o.automl import H2OAutoML
import click
import configs


@click.command()
@click.argument("dataframe_path", type=click.Path())
@click.argument("output_path", type=click.Path())
def train_and_save(dataframe_path, output_path):
    """
    Train bunch of models in h2o and save top model

    :param dataframe_path:
    :param output_path:
    :return:
    """
    h2o.init()
    train = h2o.import_file(dataframe_path)

    x = train.columns
    y = configs.TARGET

    x.remove(y)
    aml = H2OAutoML(max_runtime_secs=configs.MAX_RUNTIME_SECS, seed=configs.SEED)
    aml.train(x=x, y=y, training_frame=train)
    lb = aml.leaderboard
    print(lb.head(rows=lb.nrows))
    final_model = aml.leader
    model_path = h2o.save_model(model=final_model, path=f"{output_path}", force=True)
    print(f"Model saved to {model_path}")
    # return model_path


if __name__ == "__main__":
    train_and_save()
    # model_path = train_and_save(NEW_TRAIN_PATH, '/home/yurkoi/Desktop/project_ops/h2o_mlops/models/')
    #
    # test = h2o.import_file(NEW_TEST_PATH)
    # saved_model = h2o.load_model(model_path)
    # preds_y = saved_model.predict(test)
    # pred_test = preds_y.as_data_frame().predict
    # print(preds_y)
    # print(pred_test)
