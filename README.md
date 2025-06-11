# LLMチュートリアル@UEC

## 0.1 環境設定

サーバなど中央集約型のシステムにログインして作業するとき、それぞれのユーザごとに環境を作成する必要があります。

今回は環境を統一するために[uv](https://github.com/astral-sh/uv)というパッケージ管理ツールを使います。

サーバにログインした後、下記コマンドを実行してください。
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

その後に、 `uv -h`と入力したとき、
```
pine11% uv -h
An extremely fast Python package manager.

Usage: uv [OPTIONS] <COMMAND>
...
```
などのように出力されれば、インストール完了です。

### 参考資料
- https://docs.astral.sh/uv/
- https://zenn.dev/watanko/articles/8400df75e4aab5
- https://dev.classmethod.jp/articles/uv-unified-python-packaging-explained/
- https://qiita.com/yuya-0405/items/41c8153530d5542bee16


## 0.2 仮想環境

各研究やプロジェクトごとに、異なるバージョンのライブラリを使用したいニーズが頻繁に発生します。
あるいは、本番環境と開発環境を分けたい状況等では、他のプロジェクトに影響を与えないように、
新しく環境を構築したいです。このような場面で、実際の自分の環境とは別に、プロジェクトごとの仮想環境を作成することが一般的です。

仮想環境の設定はuvを使った場合、非常に簡単です。下記のコードを実行してください。今回はPython3.10に統一しています。
```
uv venv --python 3.10
source .venv/bin/activate
```

このとき、
```
pine11% uv venv --python 3.10
source .venv/bin/activate
Using CPython 3.10.16
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate
(llm-tutorial) pine11%
```
のように、括弧がカーソルの前に登場したら成功です。

試しに、`python`と入力すると、
```
(llm-tutorial) pine11% python
Python 3.10.16 (main, Dec  6 2024, 19:59:16) [Clang 18.1.8 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```
のようになって、`Python 3.10.16`がインストールされていることが確認できるはずです。

## 