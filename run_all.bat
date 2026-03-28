@echo off
echo Training BERT...
python src/train_bert.py

echo[
echo Training Text-CNN...
python src/train_cnn.py

echo[
echo Running RL evaluation...
python experiments/rl_evaluation.py

echo[
echo Generating learning curves...
python src/plot_learning_curves.py

echo[
echo Done! Run predict.py now
pause