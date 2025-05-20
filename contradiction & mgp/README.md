### 程式碼簡單說明

在最一開始的版本，是只靠執行的電腦完成 **MGP相似句子搜尋** 與 **矛盾三元組匹配** 的，然而，單靠一台電腦在 **矛盾三元組匹配** 的任務中速度不盡理想（太花時間）。

因此，特地寫了 ``helper.py`` 檔案，其主要功能就是透過 **gradio** 套件，讓其他機器可以透過 **gradio** 自行產生的網址，傳遞資料給執行 ``helper.py`` 的電腦，在該電腦（在此稱為 helper machine）進行運算，隨後在發送回原電腦。

補充：雖然主要的瓶頸為 **矛盾三元組匹配**，但後來連 **MGP相似句子搜尋** 也外包給 helper machine 做了。

以下舉例說明

假設以下情況
> 電腦 A 為主要電腦（負責提供使用者介面）
> 電腦 B, C 為負責矛盾的 helper machine
> 電腦 D 為負責MGP搜尋的 helper machine 

具體步驟如下
1. 先在電腦 B 與 電腦 C 上執行 ``helper.py``
    ```bash
    python helper.py --type contrad
    ```
    執行後，電腦 B 跟 電腦 C 會出現一個 public url
2. 在電腦 D 上執行 ``helper.py``
    ```bash
    python helper.py --type mgp
    ```
    執行後，電腦 D 會出現一個 public url
3. 在電腦 A 上，修改 ``interface.py`` 中的 ``machine_type``、``helper_machine_urls`` 與 ``mgp_helper_machine_url``
    ```python 
    [29] machine_type = Machine.multi_machine   # 另一個值為 single_machine
    ...
    [35] helper_machine_urls: list[str] = [ 
    [36]    # "https://xxx.gradio.live"
    [37]    "放入電腦 B 的網址",
    [38]    "放入電腦 C 的網址", ... # 如果有更多電腦也可以加上去
    [39] ]
    [40] mgp_helper_machine_url = "放入電腦 D 的網址" # "https://xxx.gradio.live"，MGP 資料庫搜尋只需要一台電腦執行即可
    ```
4. 在電腦 A 上，執行 ``gui.py``。

透過以上方法，電腦 A 便可以將任務分配給其餘的電腦。
需要注意的是，其實並不一定要將 MGP搜尋的協助電腦 跟 矛盾的協助電腦 分開（也就是可以讓電腦 D 跟電腦 B 是同一台，只是分別執行相應的程式碼而已）。 
同理，如果本身電腦夠好，也可以在電腦 A 在執行電腦 B 的任務。（也就是在同一台電腦同時執行 ``helper.py`` 跟 ``gui.py``）

若想回歸到最初版本（所有計算全靠電腦 A ），則可透過修改 ``interface.py`` 中的 ``machine_type`` 達到。

### 注意事項
* ``numpy`` 版本需下調至 ``1.26.4``

### 資料下載
* **[data](https://www.dropbox.com/scl/fo/jadxfd0noq6h6ojkzlh63/AMdz_ykissSpZTsibaEULlQ?rlkey=bp6vbma8x6x72yc8dwgrr0oed&st=86931m70&dl=0)**

