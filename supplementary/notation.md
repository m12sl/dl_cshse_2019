# Нотация


## Размерности тензоров

**Keywords: `channels_first`, `channels_last`, `NHWC`, `NCHW`, `spatial`, ...**

В глубоком обучении тензор -- это просто многомерный массив, у которого размерности имеют разный смысл. Обычно выделяют батчевую размерность, каналы и пространственные размерности.

При работе с 2d-данными (картинками) характерные размеры тензоров `[batch_size, channels, height, width]`

`batch_size` -- это батчевая размерность, `channels` -- каналы или цвета.
`width, height` -- это пространственные размерности.


Широкораспространенны два порядка размерностей:
 - channels first: `[bs, ch, h, w] ` по умолчанию в Pytorch и CUDNN
 - channels last: `[bs, h, w, ch]` по умолчанию в TF/Keras и т.д.


При работе с 1d-данными (последовательностями, словами, сигналами) характерные размеры:
`[bs, ch, steps]`, порядки осей такие же.

При работе с 3d-данными (видео, 3д сцены и т.д.) характерные размеры: `[bs, ch, depth, height, width]`.


В названиии операций принято писать вдоль какого количества пространственных размерностей они действуют:
 - MaxPool1d
 - BatchNorm2d
 - Conv3d


В статьях по компьютерному зрению по-умолчанию подразумевают использование Conv2d, и говорят про пространственные размеры ядер: 3x3-, 1х1-, 1x5- свертки -- это все про пространственные размер скользящего окна.


## RNN и последовательности

При использовании CUDALSTM порядок осей может показаться неожиданным `[steps, batch_size, channels]`, он же `major_time`.
