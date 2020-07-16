Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks [Finn et al., ICML'17]の再現実装です。

```
python main.py
```
で学習し、

```
python test.py
```
で、学習済みのモデルを用いてテストします。

動作環境としては、
[NVIDIA NGC](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch)から公開されているpytorch:20.03-py3のコンテナを使用し、また[torchmeta](https://pypi.org/project/torchmeta/#description)というライブラリを使用しています。

このリポジトリの解説記事は[ここ](https://qiita.com/gen10nal/items/204bc92de1a4147e5e18)にあります。