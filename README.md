# paper
論文の翻訳

|案|メリット|デメリット|決定|
|:--|:--|:--|:--|
|案A:xxxする| | | |
|案B:xxxする| | | |
|案C:xxxする| | | |

## xxxx会議録
日時:2018/10/xx
場所:xxx会議室
出席者:xxx,xxx,xxx,xxx

### 1.目的
* xxxxの対応を決める。
* xxxxの対応を決める。

### 2.結論
* xxxxをxxxとする。
* xxxxをxxxとする。

### 3.A.I.
* xxxxxする。(担当: xxx 期日： x/xまで)
* xxxxxする。(担当: xxx 期日： x/xまで)
* xxxxxする。(担当: xxx 期日： x/xまで)

### 4.議論
* xxxxはxxxではないか？(Aさん)
→ xxxにする。

__xxxのワークフロー__
```plantuml
@startuml
|__担当A__|
:xxxする;
|__担当B__|
:xxxする;
|#ccf|__担当C(部署X)__|
:xxxxする;
|__担当B__|
:xxxする;
fork
|__担当B__|
:xxxする;
forkagain
|#cfc|__担当D(部署Y)__|
:xxxする;
end fork
|__担当A__|
:xxxする;
@enduml
```

__開発プロセス(PFD)__
```plantuml
@startuml
agent 企画書
agent 要求仕様書
agent 設計書
agent ソースコード

(企画)-->[企画書]
[企画書]-right->(要件定義)
(要件定義)-right->[要求仕様書]
[要求仕様書]-right->(設計)
[企画書]-right->(設計)
(設計)-down->[設計書]
[企画書]-right->(実装)
[要求仕様書]-right->(実装)
[設計書]-->(実装)
(実装)-->[ソースコード]
@enduml
```

```
__\```plantuml__
@startuml
|__担当A__|
:xxxする;
|__担当B__|
:xxxする;
|#ccf|__担当C(部署X)__|
:xxxxする;
|__担当B__|
:xxxする;
fork
|__担当B__|
:xxxする;
forkagain
|#cfc|__担当D(部署Y)__|
:xxxする;
end fork
|__担当A__|
:xxxする;
@enduml
__\```__
```

__PDPC__
```plantuml
@startuml
(*)-->"xxxする"
if "A問題ができた?" then
-->[Yes]"yyyする"
if "B問題ができた?"then
-->[Yes] "mmmする"
-->"cccする"
else
-->[No] "nnnする"
-->"cccする"
endif
else
-->[No]"aaaする"
-->"bbbする"
-->"cccする"
endif
"cccする"-->(*)
@enduml
```

```plantuml
@startuml
actor A

node "社内"{
    A -up-> [xxxサーバー] : アクセス
}

node "AWS"{
    database DB
    [xxxサーバー]-right->[xxxAPI] : hoge
    [xxxAPI]-right->DB : xxxする   
}
@enduml
```

