# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .estimator import SBDEstimator
from .vocabulary import Vocabulary
from .input import train_input_fn, serve_input_fn
import argparse
import os
import sys
import tensorflow as tf


def main(argv):
    del argv

    # Load vocabulary
    vocab_filename = os.path.join(FLAGS.data_path, 'vocabulary.pkl')
    vocab = Vocabulary.load(vocab_filename)

    estimator = SBDEstimator(
        min_n=FLAGS.min_n,
        max_n=FLAGS.max_n,
        ngram_vocab=vocab.items(),
        uniq_count=FLAGS.uniq_count,
        embed_size=FLAGS.embed_size,
        embed_dropout=FLAGS.embed_dropout,
        rnn_size=FLAGS.rnn_size,
        rnn_layers=FLAGS.rnn_layers,
        use_cudnn=FLAGS.use_cudnn,
        rnn_dropout=FLAGS.rnn_dropout,
        learning_rate=FLAGS.learning_rate,
        model_dir=FLAGS.model_path,
    )

    # Run training
    # hook = tf.train.ProfilerHook(save_steps=2, output_dir=FLAGS.model_path, show_memory=True)
    train_wildcard = os.path.join(FLAGS.data_path, 'train*.tfrecords.gz')
    estimator.train(input_fn=lambda: train_input_fn(train_wildcard, batch_size=5))

    # Run evaluation
    eval_wildcard = os.path.join(FLAGS.data_path, 'valid*.tfrecords.gz')
    metrics = estimator.evaluate(input_fn=lambda: train_input_fn(eval_wildcard, batch_size=5))
    print(metrics)

    if len(FLAGS.export_path):
        estimator.export_savedmodel(FLAGS.export_path, serve_input_fn)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train and evaluate SBD model')
    parser.add_argument(
        'data_path',
        type=str,
        help='Path with TFRecord files and vocabulary')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to store model')
    parser.add_argument(
        '-export_path',
        type=str,
        default='',
        help='Path to store model')
    parser.add_argument(
        '-min_n',
        type=int,
        default=3,
        help='Minimum ngram size')
    parser.add_argument(
        '-max_n',
        type=int,
        default=4,
        help='Maximum ngram size')
    parser.add_argument(
        '-uniq_count',
        type=int,
        default=1000,
        help='Number of <UNK> vocabulary items')
    parser.add_argument(
        '-embed_size',
        type=int,
        default=50,
        help='Ngram embedding size')
    parser.add_argument(
        '-embed_dropout',
        type=float,
        default=0.01,
        help='Input ngram emmbedding dropout probability')
    parser.add_argument(
        '-rnn_size',
        type=int,
        default=64,
        help='RNN layer size')
    parser.add_argument(
        '-rnn_layers',
        type=int,
        default=1,
        help='RNN layers count')
    parser.add_argument(
        '-use_cudnn',
        default=False,
        action='store_true',
        help='Use Cudnn LSTM vs TF LSTM')
    parser.add_argument(
        '-rnn_dropout',
        type=float,
        default=0.2,
        help='RNN dropout probability')
    parser.add_argument(
        '-learning_rate',
        type=float,
        default=0.001,
        help='Learning rate')
    parser.add_argument(
        '-no_export',
        default=False,
        action='store_true',
        help='Do not export trained model')

    FLAGS, unparsed = parser.parse_known_args()
    assert os.path.exists(FLAGS.data_path) and os.path.isdir(FLAGS.data_path)
    assert not os.path.exists(FLAGS.model_path) or os.path.isdir(FLAGS.model_path)
    assert not os.path.exists(FLAGS.export_path) or os.path.isdir(FLAGS.export_path)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)



# from .hook import MetadataHook
# translit http://userguide.icu-project.org/transforms/general
# from tensorflow.contrib.learn import DynamicRnnEstimator

# Preparing
# tf.logging.set_verbosity(tf.logging.INFO)
# data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
#
# # Load vocabulary
# vocab_filename = os.path.join(data_dir, 'vocabulary.pkl')
# vocab = Vocabulary.load(vocab_filename)
#
# # Create estimator
# model_dir = os.path.join(os.path.dirname(__file__), '..', 'model')
# config = None # todo
# # {
# # _save_checkpoints_secs:3600,
# # _keep_checkpoint_max:2,
# # }
# estimator = SBDEstimator(
#     min_n=3,
#     max_n=4,
#     ngram_vocab=vocab.items(),
#     uniq_count=1000,
#     embed_size=50,
#     rnn_size=32,
#     rnn_layers=1,
#     keep_prob=0.8,
#     learning_rate=0.01,
#     model_dir=model_dir,
#     config=config,
# )
#
# # Run training
# # hook = MetadataHook(save_steps=1, output_dir=model_dir)
# train_wildcard = os.path.join(data_dir, 'train*.tfrecords.gz')
# estimator.train(input_fn=lambda: train_input_fn(train_wildcard, batch_size=5))
#
# # Run evaluation
# eval_wildcard = os.path.join(data_dir, 'valid*.tfrecords.gz')
# metrics = estimator.evaluate(input_fn=lambda: train_input_fn(eval_wildcard, batch_size=5))
# print(metrics)
#
# # Run prediction
# document = u"""
# Народный дом.
#
# Народный дом — в дореволюционной России общедоступное культурно-просветительское учреждение. Большинство народных домов до 1914 г. были государственными (например, земские и муниципальные дома попечительства о народной трезвости), однако встречались и негосударственные народные дома, построенные и финансируемые частными благотворителями. Создавались начиная с конца 1880-x гг, особенно широко — после революции 1905 года. Народные Дома существовали и существуют и по ныне в Чехии. Причем практически в каждом городе, а также в ряде стран Восточной Европы ещё до 1880 года. Например, в Карлсбаде (ныне: Карловы Вары). Гости из России могли перенять эту форму объединения растущих общественных интересов, а заодно и благозвучное название. Народный Дом — место, где общества и творческие кружки могли проводить репетиции, концерты, балы, гильдии устраивать вечера, лекции, встречи и т. п. Владели Народным Домом либо несколько гильдий, либо городские (муниципальные) власти.   Существует мнение, что непосредственными предшественниками народных домов в России были народные дома Англии. В 1887 году в Англии возник новый тип народных домов — многофункциональных учреждений, предоставлявших вечернее образование для взрослых и внешкольное — для детей. Но впервые Народные дома появились в России, а не в Англии — первый народный дом возник в 1882 году в Томске. В Петербурге первый народный дом открылся в 1883 году. Народные дома России XIX — начала XX века старались объединить все формы образовательной и досуговой деятельности. Организуя культурный досуг населения, они ставили перед собой задачу развивать внешкольное образование, бороться с неграмотностью, вести лекционную работу. В них размещались библиотека с читальней, театрально-лекционный зал со сценической площадкой, воскресная школа, вечерние классы для взрослых, хор, чайная, книготорговая лавка. При некоторых народных домах устраивались музеи, где сосредотачивались различного типа наглядные пособия, используемые при чтении лекций в процессе систематических занятий, передвижные и постоянно действующие выставки. Проектированием народных домов занимались Ф. О. Шехтель, И. А. Иванов-Шиц, П. П. Рудавский и др. видные архитекторы. В 1910—1914 А. У. Зеленко и И. П. Кондаков совместно выполнили серию проектов типовых народных домов (не реализованы). Крупнейшим народным домом вне Петербурга был Аксаковский народный дом в Уфе, начатый постройкой в 1909, с залом на 600 мест, впоследствии расширенным. В 1905—1917 строительство народных домов получило официальную поддержку как минимум муниципальных, городских властей — в надежде связать революционную активность населения, занять делом рабочую молодёжь и противодействовать массовому пьянству (соучредителями народных домов часто выступают общества трезвости). В январе 1915 Московская городская дума постановила открыть к 1 сентября 1915 двенадцать народных домов и открывать ежегодно по три новых (до 1919 включительно). Однако до октября 1917 были открыты только два учреждения. В Москве народные дома так и не стали крупными просветительскими центрами, не смогли привлечь большое число посетителей, несмотря на низкие расценки и бесплатные программы.  После революции 1917 народные дома сохранили свои функции под контролем новой власти — став рабочими клубами. Скромные размеры старых народных домов не отвечали потребностям индустриализованных городов; так, в послевоенные годы б. Введенский народный дом в Москве был перестроен до неузнаваемости в стиле позднего сталинского ампира (сейчас в здании — Дворец культуры Московского электролампового завода). В советское время название Народный дом в официальной речи использовалось редко. В 1937 арх. А. О. Таманян проектирует Ереванский Народный Дом. Проект был удостоен гран-при на парижской выставке 1937 года и реализован как Театр оперы и балета Армянской ССР.            Санкт-Петербург: Народный дом Николая II (1900, в основе здания — конструкции арх. А. Н. Померанцева, не сохранился) Опера Народного дома (1912, арх. Г. И. Люцедарский). C 1950-x гг — Ленинградский мюзик-холл Народный дом-читальня им. Нобеля на Лесном проспекте (1901, арх. Р. Ф. Мельцер) [1] Лиговский народный дом им. С. В. Паниной (1903) — ДК железнодорожников  Москва: Введенский народный дом (1903—1904, арх. И. А. Иванов-Шиц, полностью перестроен в конце 1940-х гг.) — единственный муниципальный народный дом. Кроме него, в 1914 существовало 11 независимых народных домов.  Уфа: Аксаковский народный дом (1909—1914, арх. П. П. Рудавский, достроен в 1928—1935). В здании располагается Башкирский государственный театр оперы и балета. [2]  Челябинск: Челябинский народный дом (1903). В здании располагается Челябинский государственный молодёжный театр.  Барнаул Барнаульский народный дом — ныне в здании расположена краевая филармония.  Бийск Бийский народный дом — сегодня располагается Драматический театр.  Воронеж Народный дом — разрушен во время Второй мировой войны.   Киев: Лукьяновский народный дом (1902). После 1917 — клуб трамвайщиков Троицкий народный дом (1902). С 1934 года — театр оперетты  Львов: Народный дом — одна из галицко-русских (в какой-то период украинских) организаций. Первый Народный Дом во Львове был сооружен в 1851-54 гг. Он имел название «Руський Народный Дом». Всего в Галиции до Первой мировой войны было построено более 500 Народных Домов. До настоящего времени сохранились Коломыйский, Стрыйский, Яворовский, Перемышльский (теперь территория Польши),Борщевский и некоторые другие Народные Дома. Во Львовском Народном Доме, построенном на руинах монастыря тринитариев (в 1848 году австрийский император Франц-Иосиф передал разрушенный бомбежками монастырь под строительство русской церкви), находятся воинские учреждения.  Усть-Каменогорск В бывшем здании Народного дома сегодня работает Восточно-Казахстанский областной театр драмы имени Жамбыла        Широкое распространение получили народные дома (швед. Folkets hus) в Скандинавских странах: Швеции, Дании и Норвегии, где народные дома есть во многих городах.  Название «Народный дом» (итал. Casa del Popolo) появилось также в Италии в сентябре 1893 года во время второго конгресса социалистов в городе Реджо-нель-Эмилия, в связи с учреждением нового общественного здания для кооператива в Массенцатико (городке неподалёку от Реджо). На становление этого понятия в Италии оказал влияние и соответствующий опыт соседних стран (см. ниже).  Итальянские Каза дель Пополо того времени отвечали потребностям развития народных кооперативов для совместной работы и потребления, как и всему комплексу задач, служащему культурным, благотворительным, и досугово-развлекательным нуждам вкупе с механизмами взаимопомощи. Новое дыхание и звучание это движение получило во второй половине XX века, попав под протекторат Итальянской Коммунистической партии (и отчасти других левых сил). С этого времени итальянские «Каза дель Пополо» воспринимаются как коммунистическая разновидность многостороннего общеевропейского явления, за которым закрепилось название «Общественный центр». Для коммунистического и социалистического движения Италии, — Народный дом символизирует координационный центр одновременно как для политического сплочения, так и создания модели будущего общества, низовой ячейки-ядра для социализма, которые могли бы шаг за шагом расширяться и нарастать вплоть до включения в себя местного управления, экономической жизни и гражданского общества целиком.  В этом смысле, «Каза дель Пополо» (в современной «левой» трактовке) олицетворяет надежду не только на построение «нового общества», но и формирование нового «социалистического» человека.  В Болгарии самые распространённые подобные, но с национальной спецификой общественные клубные учреждения — Читалиште, известны с 1856 года. Схожие — и в то же время национально-своеобразные разновидности народных домов известны также в опыте множества других европейских стран, как, например, Maison du peuple во Франции, Бельгии и Швейцарии (первый швейцарский народный дом итальянского типа также основан в 1899 году в городе St. Gallen), или Volkhaus в Германии, Volkshuis в Голландии и т. п. Самые известные примеры: Maison du Peuple в Брюсселе (называемый также по-фламандски — Volkshuis Brussel), Maison du Peuple в Нанси, Maison du Peuple в Клиши.  Общественный центр Дом культуры  Москва начала века / авт.-сост. О. Н. Оробей, под ред. О. И. Лобова. — М.: O-Мастеръ, 2001. — С. 367—368. — 701 с. — (Строители России, ХХ век). — ISBN 5-9207-0001-7. Рябков В. М. Антология форм праздничной и развлекательной культуры России (XVII — начало XX вв.): уч.пособие / В. М. Рябков. — Чел.акад.культуры и искусств. — Челябинск, 2006. — 706 с. Виноградов А. П. История культурно-просветительской работы в СССР − 1970. — 246 с.
# """
#
# predictions = estimator._predict(input_fn=lambda: predict_input_fn(document))
# predictions = list(predictions)[0]
# words = list(predictions['tokens'])
# classes = list(predictions['classes'])
#
# sentences = []
# sentence = []
# for word, boundary in zip(words, classes):
#     sentence.append(word.decode('utf-8'))
#     if boundary:
#         sentences.append(u''.join(sentence))
#         sentence = []
# sentences.append(u''.join(sentence))
# print(u'\n-------------------------------------------\n'.join(sentences))
