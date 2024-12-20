### 開発環境
+ Macで開発
+ Pythonはcv2,numpy,YOLOをインストールしている

### 機械学習の手順
（ファイルパスは適宜変更）
1. スローモーションで撮影した映像をcreating_dataset.pyファイルを実行して0.5秒区切りで画像として出力（新しくフォルダがつくられる）
2. [CVAT](https://www.cvat.ai/)で新たにタスクを作成する。ラベルは「bottle」選択方法は長方形（rectangle）を選択。画像をアップロードする。
3. ペットボトルを長方形で囲む。ファイル名に数字が含まれていたことが原因かもしれないが、画像が映像の順番になっていないかもしれないので注意。
4. YOLOv8 Detection 1.0 でエクスポート
5. zipファイルを解凍し、フォルダを開くと中にlabelsフォルダが含まれる
6. 下のようなファイル構成にする（trainフォルダとvalフォルダにはそれぞれ対応する画像ファイルとtxtファイルを入れておく）
<pre>
  .
  └── dataset
      ├── dataset.yaml
      ├── train
      │   ├── images
      │   └── labels
      └── val
          ├── images
          └── labels
</pre>
8. dataset.yamlには学習時により多くの学習データを作るために変更する値を指定する
9. ターミナル上で機械学習の方法.txtに書かれたコマンドを実行
10. runsというフォルダが作られるのでその中のbest.ptが学習済みモデルとなる

※必要があり次第追記します
