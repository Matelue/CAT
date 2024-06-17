# Data preprocessing

数据处理分为以下几个步骤：

下载数据 -> 转换成kaldi格式 -> 文本正则化 -> 生成词典 -> 特征提取

以下，以波兰语为例，详细讲解：

```shell
lang=pl
work_space=/mnt/workspace/saier/cat-test/egs/commonvoice
data_space=$work_space/data
dict_dir=$work_space/dict/$lang
data_dir=$data_space/$lang
wav_dir=$data_space/cv-corpus-11.0-2022-09-21/$lang
```

## Stage -1: 下载数据

从 [CommonVoice 官网](https://commonvoice.mozilla.org/zh-CN/datasets) 下载数据并解压。

```bash
wget -c https://mozilla-common-voice-datasets.s3.dualstack.us-west-2.amazonaws.com/cv-corpus-11.0-2022-09-21/cv-corpus-11.0-2022-09-21-${lang}.tar.gz
tar -xvf cv-corpus-11.0-2022-09-21-${lang}.tar.gz
```

## Stage 0: 生成 kaldi 格式的数据

从解压后的csv文件中提取音频路径和转录文本，生成 kaldi 格式的数据。这里只需要生成 `text` 和 `wav.scp` 两个文件。

```bash
cd $work_space
mkdir -p data/$lang
for s in dev test train validated;do
    cd $work_space
    mkdir -p $data_dir
    file="$src/cv-corpus-11.0-2022-09-21/$lang/$s.tsv"
    [ ! -f $file ] && {
    echo "No such file $file"
    exit 1
    }
    # 打开file文件，去除句子后缀，只保留句子ID，临时保存为uid.tmp
    cut <$file -f 2 | tail -n +2 | xargs basename -s ".mp3" >$d_set/uid.tmp
    # 将句子路径输出为path.tmp
    cut <$file -f 2 | tail -n +2 | awk -v path="$wav_dir/cv-corpus-11.0-2022-09-21/$lang/clips" '{print path"/"$1}' >$d_set/path.tmp
    # 将句子ID与对应路径拼接到同一文件并输出到wav.scp
    paste $d_set/{uid,path}.tmp | sort -k 1,1 -u >$d_set/wav.scp
    # 将句子路径输出为text.tmp
    cut <$file -f 3 | tail -n +2 >$d_set/text.tmp
    # 将句子ID与对应文本拼接到同一文件并输出到text
    paste $d_set/{uid,text}.tmp | sort -k 1,1 -u >$d_set/text
    # 删除临时文件
    rm -rf $d_set/{uid,text,path}.tmp
done
```

代码中的变量`language_list`为所要处理的语言的ID列表，`d_set`为处理后的数据存放路径，`file`为原始转录文本路径；该阶段主要实现根据音频数据及路径生成`text`和`wav.scp`。
`text` 中的数据格式如下所示:

```
$ head -2 data/pl/dev/text
common_voice_pl_100540	same way you did
common_voice_pl_10091129	by hook or by crook
```

`wav.scp` 中的数据格式如下所示:

```
$ head -2 data/pl/dev/text
common_voice_pl_20551620	/mnt/workspace/CAT/egs/commonvoice/data/cv-corpus-11.0-2022-09-21/pl/clips/common_voice_pl_20551620.mp3
common_voice_pl_20594755	/mnt/workspace/CAT/egs/commonvoice/data/cv-corpus-11.0-2022-09-21/pl/clips/common_voice_pl_20594755.mp3
```

## Stage 1: 从训练集中去除测试集和验证集

代码如下：

```bash
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  # By default, I use validated+train as the real training data
  # ... but we must exclude the dev & test from the validated one.
  echo "stage 1: Exclude the dev & test from the train set"
  d_train="$data_dir/excluded_train"
  mkdir -p $d_train
  for file in wav.scp text; do
    cat $data_dir/{validated,train}/$file |
        sort -k 1,1 -u >$d_train/$file.tmp
    for exc_set in dev test; do
        python local/expect.py \
          $d_train/$file.tmp \
          --exclude $data_dir/$exc_set/$file \
          >$d_train/$file.tmp.tmp
        mv $d_train/$file.tmp.tmp $d_train/$file.tmp
    done
    mv $d_train/$file.tmp $d_train/$file
  done
  rm -rf $data_dir/{validated,train}
  echo $lang 'Text done'
fi
```
## Stage 2: 文本正则化

去除转录文本`text`中的特殊符号如标点符号、外来语字符，因为非语言符号无法被G2P模型识别以生成词典。

```bash
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
  echo "stage 2: Text Normalization"
  # Text Normalization
  for set in dev test excluded_train; do
    	paste $data_dir/$set/text > $data_dir/$set/text.bak
    	cut <$data_dir/$set/text.bak -f 2- | \
        sed -e 's/`/ /g; s/¨/ /g; s/~/ /g; s/=/ /g' \
         -e 's/|/ /g; s/°/ /g; s/[-]/ /g; s/[―]/ /g; s/,/ /g; s/[;]/ /g; s/:/ /g; s/!/ /g; s/¡/ /g; s/?/ /g; s/[¿]/ /g; s/′/ /g; s/‐/ /g; s/´´/ /g' \
         -e 's/[.]/ /g; s/·/ /g; s/‘/ /g; s/’/ /g; s/"/ /g; s/“/ /g; s/”/ /g; s/«/ /g; s/»/ /g; s/≪/ /g; s/≫/ /g; s/[{]/ /g; s/„/ /g; s/−/ /g; s/‑/ /g' \
         -e 's/[}]/ /g; s/®/ /g; s/→/ /g; s/ʿ/ /g; s/‧/ /g; s/ʻ/ /g; s/ ⃗/ /g; s/‹/ /g; s/›/ /g; s/_/ /g; s/ʽ//g; s/￼￼/ /g; s/m̪/m/g; s/ː/ /g; s/ﬁ/fi/g; s/ﬂ/fl/g' \
         -e 's/[–]/ /g; s/…/ /g' \
         -e "s/\// /g; s/#/ /g; s/&/ & /g; s/´/'/g; s/''/ /g; s/^[']*/ /g; s/[']*$/ /g; s/ '/ /g; s/' / /g; s/\[/ /g; s/\]/ /g" \
         -e 's/&/ /g;s/(/ /g;s/)/ /g;s/\\/ /g;s/—/ /g;s/，/ /g;s/！/ /g;' | \
         sed -e 's/[ ][ ]*/ /g; s/^[ ]*//g; s/[ ]*$//g' | \
         python -c "import sys; print(sys.stdin.read().lower())" > data/$lang/$set/text.trans.tmp
      	cut <$data_dir/$set/text.bak -f 1 > $data_dir/$set/text.id.tmp
        paste $data_dir/$set/text.{id,trans}.tmp > $data_dir/$set/text
        cat $data_dir/$set/text | sed -e 's/^[	]*//g' | grep -v "^$" > $data_dir/$set/text_new
        mv $data_dir/$set/text_new $data_dir/$set/text
        rm -rf $data_dir/$set/text.{id,trans}.tmp
    done
  echo $lang 'Text normalization done'
fi
```
每个语言的文本中的特殊符号不一样，因此需要先根据文本打印出所有字符，将字符集输入给G2P模型，统计不能识别的字符，再调用`text_norm.sh`进行去除。
注：
    （1）对于非语言符号，在脚本中添加待删字符即可使用脚本去除；注意有些字符如`$`、`\`等需在该符号前面加转义字符`\`，否则命令无法被正确执行。
    （2）对于外来语字符，由于其具有发音信息，只删除某个词可能会影响模型训练，因此需直接删除包含外来语的整条句子。
    （3）为避免误删，删除前建议对`text`和`wav.scp`进行备份，所有正则化操作只对`text`进行。

## Stage 3: 词表生成

统计转录文本中的词，生成唯一的词表，一个词为一行，格式如下所示:

```
$ head -3 $dict_dir/word_list
a
aaron
ababa
```

代码如下：
```bash
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
	mkdir -p $dict_dir
  	text_file="$data_dir/*/text"
  	cat $text_file | awk -F '\t' '{print $NF}'  | sed -e 's| |\n|g' | grep -v "^$" | sort -u -s > $dict_dir/word_list
  	echo $lang 'Word list done'
  	python local/char_list.py $dict_dir/word_list
  	echo $lang 'character list done, please check special tokens in character list, confirm Text normalization is correct.'
fi
```
其中file为转录文本的路径，注意检查是否正确。

## Stage 4: 词典生成

使用去除特殊符号后的转录文本作为G2P模型的输入来生成词典，因此需先下载G2P模型，下载网址https://github.com/uiuc-sst/g2ps；生成的词典中每一行为词到音素的一一对应，格式如下所示：

```
$ head -3 $dict_dir/lexicon.txt
a	ə
aaron	a a r o n
ababa	a b ə b ə
```

代码如下：
```bash
if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
  # Generating lexicon and lexicon correction
  echo "stage 4: G2P Conversion, generating lexicon"
  # 生成词典
  bash data/$lang/lexicon.sh
  # 去除词典中的特殊符号
  sed -i 's/ː//g; s/ˈ//g; s/ʲ//g; s/[ ][ ]*/ /g; s/^[ ]*//g; s/[ ]*$//g' dict/$lang/phone.txt
  cat dict/$lang/lexicon.txt | awk '{print $1}' > dict/$lang/word.txt
  paste dict/$lang/{word,phone}.txt > dict/$lang/lexicon_new.txt
  mv dict/$lang/lexicon_new.txt dict/$lang/lexicon.txt
  rm -rf dict/$lang/{lexicon_new,word,phone}.txt
  echo $lang 'Lexicon done'
fi
```
由于所使用的G2P模型有一定错误率，可能会生成错误符号，因此生成词典后还需对词典进行后处理，如删除多余符号，拆分双音素为单音素等，具体每个语言的处理需参考http://www.isle.illinois.edu/speech_web_lg/data/g2ps/ 中对应语言的词典。

