# Решаем большую задачу

**План действий:**

1. Бейзлайн, сколь угодно грязный. (Если это кеггл, то сделать первую посылку). На этом этапе надо понять как устроены входные и выходные данные и убедиться, что получается формировать корректные посылки.
2. Переформатировать бейзлайн под себя. Здесь вы начинаете писать `train.py` с параметрами удобными для запуска.
3. Улучшать решение и проверять идеи. Идеи берутся из EDA, пристального вглядывания и чтения форумов.


## Запуск на сервере

Для загрузки кода на сервер удобно настроить в pycharm автоматическую синхронизацию с удаленной машиной по ssh. 
На `*nix` машинах можно грузить файлы с помощью rsync:
```bash
rsync -Pathz ./*py ubuntu@ip:~/very_project/
```

Скрипты запускайте в `tmux` [Шпаргалка по tmux](https://habr.com/ru/post/126996/), тогда их выполнение не прервется при прерывании коннекта.
Храните данные на `/data` (разделе, который не сбросится при остановке инстанса), логи и чекпоинты складывайте в отдельной для каждого запуска папке.

**Проброс портов**

Пусть на сервере в tmux запущен tensorboard на 6006 порту
```
CUDA_VISIBLE_DEVICES= tensorboard --logdir=./path/to/logs/root --port 6006
```
Чтобы смотреть логи локально на http://localhost:7007 сделайте проброс портов:
```
ssh -L 7007:localhost:6006 ubuntu@very_ip
```



**Не забывайте о запущенной машине! По окончании расчетов гасите инстанс**


## Хорошие практики

1. Сначала пишите минимально параметризованное решение, лучше чаще итерироваться и переписывать код, чем погрязнуть в проектировании.
2. Выносите отлаженный и переиспользуемый код в отдельные файлы. Подготовку данных в `data.py`, код моделей в `models.py`, утилиты в `utils.py` и т.д.
3. Добавьте мониторинг. [tensorboardX](https://github.com/lanpa/tensorboardX) -- хороший выбор.
4. Проверяйте производительность кода. `tqdm` в коде, `watch -n1 nvidia-smi` и `htop` в консоли. Чем быстрее вы получаете отклик, тем проще проверять идеи.



## Трюки для сверточных сетей и классификации

https://arxiv.org/abs/1812.01187




### Некоторые трюки на pytorch

**Find LR**
```python
flr_logs = []
for batch in tqdm(dataloader):
    t = step / total_steps
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    
    lr = np.exp((1 - t) * np.log(lr_begin) + t * np.log(lr_end))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.zero_grad()
    out = model(x)
    loss = F.binary_cross_entropy_with_logits(out, y)
    loss.backward()
    optimizer.step()

    probs = torch.sigmoid(out).cpu().data.numpy()
    flr_logs['loss'].append(loss.item())
    flr_logs['lr'].append(lr)
    step += 1

fig = plt.figure()
plt.plot(flr_logs['lr'], flr_logs['loss'])
plt.xscale('log')
plt.grid()
self.train_writer.add_figure('loss_vs_lr', fig, global_step=self.global_step)
```


**Cyclic Learning Rate**

```python
t = self.global_step / cycle_len - self.global_step // cycle_len
lr = lr_min + np.arccos(np.cos(2.0 * np.pi * t)) / np.pi * (lr_max - lr_min)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

**Decoupled Weight Decay**
```python
# в оптимизаторе поставить weight_decay=0.0 и применить отдельно
for name, tensor in model.named_parameters():
    if 'bias' in name:
        continue
    tensor.data.add_(-weight_decay * lr * tensor.data)
```

**mixup**
```python
bs = x.size(0)
alpha = torch.from_numpy(np.random.beta(mixup, mixup, bs).astype(np.float32)).to(device)
rolled = torch.from_numpy(np.roll(np.arange(bs), 1, axis=0))
x = x * alpha[:, None, None, None] + x[rolled, ...] * (1.0 - alpha[:, None, None, None])
y = y * alpha[:, None] + y[rolled, ...] * (1.0 - alpha[:, None])
```

