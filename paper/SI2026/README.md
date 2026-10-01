# 計測自動制御学会システムインテグレーション部門講演会用LaTeX clsファイル

`sice-si.cls`は、計測自動制御学会システムインテグレーション部門講演会用のLaTeX clsファイルです。

## 概要

このクラスファイルは、計測自動制御学会システムインテグレーション部門講演会の予稿のためのLaTeXクラスファイルです。pLaTeX、upLaTeXはLaTeXの処理系の更新で今後使用できなくなる可能性があるため、LuaLaTeXに対応したクラスファイルを作成しました。

## 特徴

- LuaLaTeXとupLaTeXに対応していますが、特別な事情がなければLuaLaTeXを使用してください。
- 予稿執筆に必要なスタイルを定義しています。
- `reference.bib`に参考文献を記述することで、自動的に規定に沿った参考文献リストを作成します。
- 従来どおりthebibliography環境を使用することもできますが、bibLaTeXの使用を推奨します。
- pdfのバージョンは1.4に設定しています。

## 使い方

`SICE-SI_sanmple.tex`を参考にしてください。
レシピは`.latexmkrc`に記述してあり、`latexmk path/to/texfile`コマンドでコンパイルできます。uplatexを使用する場合は、`.latexmkrc`のlualatexの定義部分をコメントアウトし、uplatexの定義部分をコメントアウトを解除してください。

## 動作環境
- TeX Live 2022
- TeX Live 2023
- overleaf
- cloudlatex
- docker image(ghcr.io/being24/latex-docker)

で動作確認しています。