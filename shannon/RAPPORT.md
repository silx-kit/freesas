# Détermination de Dmax par échantillonnage de Shannon — évaluation

Évaluation de la méthode de **De Caro, Alberga, Scattarella, Sibillano, Mangini, Zhou,
Stoll & Giannini**, *Shannon sampling based approach for the structural solution of
nano-objects by laboratory and synchrotron SAXS data*, J. Appl. Cryst. (2026) **59**,
1170–1183 (+ informations supplémentaires), face au réseau de neurones dense de
`freesas.dnn`.

Base : **5416 courbes expérimentales BM29** (`sandbox/2025`) avec deux références
indépendantes — BIFT et GNOM/ATSAS — plus 517 courbes synthétiques à vérité terrain
exacte, l'apoferritine SASDFN8 (SASBDB) et les trois jeux de test du dépôt.

freesas 2026.3.0 · ATSAS 4.1.4 · python 3.13 · numpy 2.5.2 · 2 septembre 2026

---

## Conclusion

**La méthode publiée, telle qu'elle est spécifiée, n'apporte rien sur `Dmax = 3·Rg`** —
l'heuristique déjà présente dans `auto_bift`. Sur le juge expérimental le moins biaisé
elle est à égalité (11,4 % contre 11,0 %) et ne le bat que sur 50,6 % des courbes ; sur
le banc synthétique, 21 % contre 17 %.

**La cause est structurelle, pas numérique.** Les résidus `f₁` et `f₂` ne présentent pas
de minimum mais un palier : au-delà du vrai `Dmax`, les sommes de Shannon reproduisent
les mêmes moments. Chercher l'argmin d'un palier est mal posé, et la fenêtre
`[2,5 ; 3,5]·Rg` fait tout le travail restant.

**Le résultat le plus important est ailleurs, et il relativise toute la question.** Sur
5416 courbes expérimentales, BIFT et GNOM s'accordent sur `Rg` à **5 %** et se
contredisent sur `Dmax` à **22 %** (§6.1). Les implémentations ne sont pas en cause —
freesas et ATSAS donnent le même `Rg` à 1,3 %. C'est `Dmax` qui est intrinsèquement mal
contraint par des données SAXS. **Chercher un estimateur de `Dmax` à 5 % n'a pas de sens
quand les deux méthodes de référence ne se départagent pas à mieux que 22 %.**

Sur le juge expérimental le moins biaisé — les courbes où BIFT et GNOM concordent —
Shannon publié (11,4 %) et `3·Rg` (11,0 %) sont **à égalité**, et Shannon ne bat le témoin
que sur 50,6 % des courbes. Le verdict du banc synthétique est donc confirmé par un
chemin indépendant.

**J'ai retiré ma variante du coude** : elle est la pire des quatre sur données réelles
(§6.6). Son biais croît avec la taille de l'objet, de −17 % à −64 %, donc aucune
calibration constante ne le corrige. Le mécanisme du palier reste exact, ma façon d'en
détecter l'entrée ne résiste pas aux données.

**Le réseau n'estime pas `Dmax` mais le `Dmax` de BIFT**, dont il a appris le biais
(§6.2, vérifié arithmétiquement). C'est un substitut rapide de BIFT, à lire comme tel.
Son avantage structurel solide est ailleurs : il ne demande aucun fit de Guinier, et
domine largement là où il n'y a pas de région de Guinier exploitable (12,5 % contre 47 %
pour Shannon, §6.4).

**Un avantage de Shannon que je n'avais pas anticipé** : le formalisme est covariant en
unité — `Dmax` sort dans l'inverse de l'unité de `q`, quelle qu'elle soit — là où le
réseau est lié à une grille figée en nm⁻¹ et répond faux sans le signaler si l'unité est
mauvaise. Face aux unités mixtes du SASBDB, c'est structurellement en faveur de Shannon
(§3.1). La détection d'unité elle-même est traitée au §3.3 : un simple seuil sur `q_max`
suffit (18/18), le contre-test par le réseau n'apporte rien.

**Une limite dure à connaître** : la méthode exige que le premier canal de Shannon soit
mesuré, soit `Dmax ≤ π/q_min` — 36,9 nm sur ce montage (§6.3). Au-delà, elle perd le
terme dominant de sa série alternée et se dégrade brutalement.

---

## 1. Tableau de comparaison — banc synthétique

> Le verdict qui compte est celui du **§6, sur données expérimentales**. Cette section
> porte sur des formes analytiques : la vérité terrain y est exacte, mais les formes ne
> sont pas représentatives du BioSAXS. Les deux se lisent ensemble, et leurs écarts sont
> commentés au §6.7.

52 courbes de validation indépendantes — tailles, paramètres de forme, graines de bruit
et populations polydisperses que la mise au point n'a jamais vues. Les 10 cas où le fit
de Guinier lui-même échoue sont écartés : aucun estimateur ancré sur Guinier ne peut y
fonctionner, et on rejetterait ces données en production.

| Estimateur | médiane | p90 | max | coût |
|---|---:|---:|---:|---:|
| **coude** (proposé, *rétracté au §6.6*) | **5,16 %** | 26,5 % | 54,5 % | 19 ms |
| réseau de neurones | 8,75 % | 33,0 % | 35,5 % | **0,23 ms** |
| `3·Rg` (témoin) | 16,75 % | 26,7 % | 32,0 % | 0 ms |
| Shannon publié | 21,38 % | 36,8 % | 39,6 % | 18 ms |

Le témoin `3·Rg` est l'élément décisif : un estimateur qui ne le bat pas ne mérite pas
d'être intégré. La méthode publiée ne le bat pas. Sur le banc principal (455 courbes,
bruit 0 à 20 %) la coïncidence est frappante : **18,32 % pour Shannon contre 18,31 %
pour `3·Rg`**.

Détail du banc principal par niveau de bruit (médiane / p90 de |erreur relative|) :

| bruit | shannon | shannon_ex¹ | dnn | 3·Rg |
|---|---:|---:|---:|---:|
| 0 % | 8,55 / 21,9 | 7,43 / 30,0 | 18,55 / 40,7 | 16,95 / 23,1 |
| 2 % | 18,20 / 42,2 | 8,54 / 33,2 | 18,44 / 41,2 | 18,28 / 27,4 |
| 5 % | 20,89 / 48,7 | 7,77 / 29,4 | 18,21 / 41,3 | 18,32 / 29,3 |
| 10 % | 20,04 / 45,3 | 10,99 / 28,4 | 17,91 / 41,3 | 18,41 / 29,3 |
| 20 % | 19,13 / 50,6 | 7,59 / 23,4 | 17,83 / 40,7 | 18,80 / 30,9 |

¹ `shannon_ex` = la méthode publiée alimentée par les `Rg` et `I(0)` **exacts** au lieu
du fit de Guinier. Ablation : elle isole l'étape Shannon de l'erreur d'ancrage. L'écart
entre les deux colonnes (18 % contre 8 %) est le premier indice du problème de
conditionnement.

> **Référence utile** : sur la même sphère de 10 nm, BIFT lui-même donne 10,18 nm, soit
> +1,8 %. Aucun estimateur rapide n'approche cette précision. Ils ne peuvent servir qu'à
> accélérer ou robustifier BIFT, jamais à le remplacer.

---

## 2. Pourquoi la méthode publiée échoue

### 2.1 Les résidus ont un palier, pas un minimum

La méthode cherche le `Dmax` pour lequel les moments reconstruits sur les canaux de
Shannon coïncident avec ceux du tracé de Guinier. Mesuré sur l'apoferritine SASDFN8
(`nmax` = 11), `Rg_NS(Dmax)` **sature** :

| Dmax d'essai (nm) | Rg reconstruit (nm) | dRg/dDmax | I0 reconstruit | amplification ∂lnDmax/∂lnRg |
|---:|---:|---:|---:|---:|
| 10,50 | 4,8635 | 0,4932 | 4,487e+04 | 0,94 |
| 11,00 | 5,0370 | 0,3470 | 4,803e+04 | 1,32 |
| 11,50 | 5,1450 | 0,2160 | 4,998e+04 | 2,07 |
| 12,00 | 5,2042 | 0,1183 | 5,069e+04 | 3,67 |
| 12,50 | 5,2140 | 0,0198 | 5,020e+04 | **21,10** |
| 13,00 | 5,2132 | −0,0016 | 5,052e+04 | **−248** |
| 14,00 | 5,2253 | −0,0000 | 5,065e+04 | **−14 747** |
| 16,00 | 5,2206 | 0,0538 | 5,060e+04 | 6,06 |
| 19,00 | 5,2471 | 0,1918 | 5,098e+04 | 1,44 |

Au-delà de 12 nm, `Rg_NS` reste bloqué entre 5,19 et 5,25 nm et `I0_NS` à 5,05·10⁴ : la
courbe ne porte plus aucune information sur `Dmax`. Résoudre
`Rg_NS(Dmax) = Rg_Guinier` revient à intersecter cette courbe avec une horizontale :

* avec le `Rg` de l'article (5,237 nm) → une bonne solution à 12,6 nm **et cinq
  parasites** au-delà ;
* avec le `Rg` que rend `auto_guinier` (5,346 nm, soit +2,1 %) → **aucune solution**,
  l'argmin part alors n'importe où (16,6 nm mesuré, +31 %).

C'est physiquement attendu : sous-estimer `Dmax` replie la série alternée et fausse les
moments, la surestimer est inoffensif — on ajoute des canaux et la série converge vers
les mêmes moments.

Conséquence pratique brutale : sur une sphère synthétique **sans bruit**, un écart de
**+0,07 %** sur `Rg` déplace l'estimation de **21 %**.

### 2.2 L'équation 12 ne diagnostique rien

Le critère d'auto-cohérence de l'article — les deux minima doivent concorder à 1 % près,
présenté comme une validation du tracé de Guinier — est satisfait dans **76 %** des 455
cas, et l'erreur médiane est **identique** selon qu'il l'est ou non : **18,26 % contre
18,60 %**. Logique une fois le palier compris : le critère compare deux positions
quasi aléatoires sur un plateau ; leur coïncidence occasionnelle ne prouve rien.

### 2.3 La fenêtre de balayage fait le travail

Test de falsification : on déplace la fenêtre de recherche et on regarde si la réponse
suit. Elle suit. Format `Dmax estimé / centre de la fenêtre` :

| Cas | vrai | [2,0 ; 3,0] | [2,5 ; 3,5] | [3,0 ; 4,0] | [3,5 ; 4,5] |
|---|---:|---|---|---|---|
| sphère | 8,0 | 8,06 / 7,94 | 10,60 / 9,53 | 11,86 / 11,12 | 11,80 / 12,71 |
| sphéroïde ε = 2 | 10,0 | 8,19 / 6,82 | 8,61 / 8,19 | 8,61 / 9,55 | 9,61 / 10,91 |
| core-shell | 10,0 | 11,82 / 10,61 | 12,95 / 12,74 | 13,24 / 14,86 | 14,86 / 16,98 |
| parabole P(r) | 12,0 | 13,32 / 11,83 | 13,20 / 14,19 | 17,63 / 16,56 | 17,66 / 18,92 |

L'estimation reste collée au centre de la fenêtre, quelle que soit la vérité. La fenêtre
`[2,5 ; 3,5]·Rg` est centrée sur `3·Rg` — d'où l'égalité de performance avec le témoin.

À noter que l'article autorise lui-même « d'autres intervalles adaptés si nécessaire »,
et que son `Dmax` publié pour l'apoferritine (126,4 Å = 2,42·`Rg`) tombe **hors** de la
fenêtre annoncée.

### 2.4 Ce n'est pas un défaut d'implémentation

Vérifié avant de conclure :

* **Les équations 3 et 5 sont correctes.** Au `Dmax` exact et sur grille fine, `I(0)` et
  `Rg` sont reproduits à 0,01–0,5 % près pour les quatre formes testées.
* **Les corrections de troncature (éq. 10–11) fonctionnent.** Sur la parabole à `nmax`=5,
  l'erreur sur `I(0)` passe de 1,95 % à 0,41 %.
* **Sur l'apoferritine au `Dmax` publié**, les sommes donnent `I0` à −1,3 % et `Rg` à
  −2,4 % — et ces −2,4 % sont exactement l'écart entre mon `Rg` de Guinier et le leur.
* **Les paramètres libres ont été balayés** (`sensitivity.py`) : `nmax` de 4 à 20,
  largeur de lissage de `f₁`/`f₂` de 1 à 401 points, paramètres du filtre `q`-adaptatif
  (`m`, `α`, `q̄₀`), largeur de fenêtre. Le meilleur réglage trouvé (3,9 % médian) s'est
  révélé être précisément celui qui rend l'estimateur dépendant de la fenêtre — cf. §2.3.

---

## 3. Ce que l'évaluation révèle sur le réseau de neurones

Le réseau tient bien la comparaison — 8,75 % d'erreur médiane, 0,23 ms par appel, et une
remarquable stabilité : sur une sphère de 10 nm il rend 10,52 à 10,79 nm quelle que soit
la plage de `q` fournie (de 0,4 à 12 nm⁻¹). Il porte cependant une contrainte
structurelle qui n'est pas déclarée.

### 3.1 Le réseau est lié à une échelle absolue, la méthode de Shannon non

`freesas.dnn.preprocess()` interpole sur une grille **figée** de 1024 points dans
[0 ; 4] nm⁻¹, avec `left=1` et `right=0`. Le réseau a donc besoin d'une échelle absolue
correcte en entrée. Le formalisme de Shannon, lui, est **covariant** : `Dmax` sort dans
l'inverse de l'unité de `q`, quelle qu'elle soit.

C'est visible directement sur `SASDFX7.dat`, dont l'en-tête déclare
`REMARK 265  q : defined in inverse Angstroms` — remark que `load_scattering_data()` ne
lit pas :

| Interprétation | q (natif) | Rg Guinier | Rg réseau | Dmax réseau | Dmax coude | Dmax Shannon |
|---|---|---:|---:|---:|---:|---:|
| brut traité comme nm⁻¹ | 0,003 – 0,370 | 22,828 | 2,541 | 8,52 | 61,01 | 71,57 |
| converti Å⁻¹ → nm⁻¹ | 0,032 – 3,696 | 2,283 | 1,975 | 5,66 | 6,10 | 7,10 |

Le coude et Shannon passent **exactement d'un facteur 10** (61,01 → 6,10 ; 71,57 → 7,10) :
ils ne font que changer d'unité. Le réseau passe de 8,52 à 5,66, sans rapport de 10 —
sa réponse sur l'entrée mal dimensionnée est simplement fausse, et rien ne le signale.
Coïncidence instructive : avec la mauvaise unité, son `Rg` (2,541) tombait plus près de
la vraie valeur (2,283 nm) qu'avec la bonne (1,975).

**Face à la difficulté des unités mixtes du SASBDB, c'est un avantage net pour Shannon
et un risque réel pour le réseau.**

### 3.2 Un domaine de validité en taille d'objet — partiellement infirmé

> ⚠️ **La limite haute annoncée ci-dessous n'existe pas sur données réelles** : au-delà de
> 40 nm le réseau est en fait le meilleur des quatre estimateurs (§6.7). Mes sphères
> monodisperses à bords francs ne sont pas représentatives du BioSAXS. Le **plancher aux
> petits objets**, en revanche, est confirmé et aggravé par les données expérimentales.

Indépendamment des unités, mesuré sur des courbes synthétiques où l'échelle est connue
sans ambiguïté (sphères, `qmax` = 4 nm⁻¹, bruit 5 %) :

| Dmax vrai | Rg vrai | Rg réseau | Dmax réseau | écart |
|---:|---:|---:|---:|---:|
| 3,0 | 1,16 | 1,15 | 4,47 | +49 % |
| 8,0 | 3,10 | 3,13 | 8,50 | +6 % |
| 20,0 | 7,75 | 8,04 | 26,47 | +32 % |
| 35,0 | 13,56 | 12,35 | 42,07 | +20 % |
| **60,0** | 23,24 | **0,96** | **0,16** | **−99,7 %** |
| **100,0** | 38,73 | **2,09** | **5,96** | **−94 %** |

Sur ces formes-là, le domaine utile est `Dmax` ∈ [5 ; 40] nm à `qmax` = 4 nm⁻¹, et le
mécanisme de l'échec est **le même** que celui du §3.1 : dès que la structure informative
de la courbe est comprimée dans le bas de la grille — parce que l'objet est gros, ou
parce que l'unité est fausse — le réseau sort de sa distribution d'entraînement. Une
seule cause, deux déclencheurs. Le réseau produit aussi des `Dmax` négatifs sur
certaines courbes à très faible contraste.

Mais la première partie de ce diagnostic ne survit pas aux données expérimentales : les
gros objets réels ne ressemblent pas à des sphères à bords francs, et le réseau les
traite très bien (§6.7). Ce qui reste vrai, c'est que le réseau échoue quand la courbe
sort de sa distribution d'entraînement — ce qui n'est pas la même chose qu'une limite en
taille.

Les deux autres jeux de test du dépôt, sans ambiguïté d'unité, ne posent pas de
problème :

| Fichier | plage q (nm⁻¹) | Rg Guinier | Rg réseau | Dmax réseau | Dmax coude |
|---|---|---:|---:|---:|---:|
| bsa_005_sub.dat | 0,028 – 4,53 | 2,914 | 2,998 | 9,30 | 8,46 |
| SASDF52.dat | 0,035 – 4,94 | 3,127 | 3,294 | 10,61 | 10,12 |

(`bsa_005_sub.dat` déclare dans son en-tête `AutoRg: Rg = 2.9802` — le réseau donne
2,998, `auto_guinier` 2,914.)

**Deux correctifs à faire dans `dnn.py`, indépendamment de la suite :**

1. vérifier que la plage de `q` couvre la grille d'entraînement, et refuser sinon ;
2. contrôler le signe et l'ordre de grandeur de la sortie.

Ni l'un ni l'autre n'est présent aujourd'hui.

### 3.3 Détecter l'unité : ce qui marche et ce qui ne peut pas marcher

Le SASBDB sert les deux conventions et le fichier le dit rarement. **Aucun critère fondé
sur la forme de la courbe ne peut trancher** : sous `q → 10q`, tous les produits sans
dimension sont invariants — `q·Rg`, `q·Dmax`, le taux de suréchantillonnage, le nombre
de canaux de Shannon. Il faut une échelle absolue venue d'ailleurs.

Deux sources testées (`unit_detect.py`) :

| Test | Fichiers réels | Synthétiques (14 cas, 2 conventions) |
|---|---:|---:|
| magnitude de `q_max` (seuil 1,2) | **4/4** | **14/14** |
| contre-test par le réseau | 4/4 | 12/14 |

Le contre-test par le réseau — comparer son `Rg` au `Rg` de Guinier, qui est covariant,
sous les deux hypothèses — fonctionne mais échoue exactement là où le réseau sort de son
domaine (sphère de 40 nm), et sa marge de décision n'est pas fiable (1,0× à 1,2× sur les
échecs, mais aussi 1,2× sur une décision correcte). **Il n'apporte rien sur le simple
seuil de magnitude.**

Zone ambiguë du seuil, en prenant pour plages crédibles 0,08–1,2 Å⁻¹ et 0,8–12 nm⁻¹ :

| `q_max` écrit dans le fichier | verdict |
|---|---|
| ≤ 0,5 | Å⁻¹ sans ambiguïté |
| 0,8 – 1,2 | **ambigu** |
| ≥ 1,5 | nm⁻¹ sans ambiguïté |

La bande d'incertitude est donc étroite. Ordre de résolution recommandé :

1. **lire l'en-tête** quand il déclare l'unité (`REMARK 265 q : defined in inverse
   Angstroms` du SASBDB) — autoritatif, et actuellement ignoré par `sasio.py` ;
2. sinon, **seuil sur `q_max`** ;
3. dans la bande ambiguë, **refuser de deviner** et exiger `-u/--unit`, qui existe déjà
   dans les CLI.

Les quatre jeux examinés confirment le point 1 : seul `SASDFX7.dat` porte la déclaration,
`SASDFN8.dat` et `SASDF52.dat` n'ont aucun en-tête exploitable.

---

## 4. Une piste qui semblait marcher : chercher le coude

> ⚠️ **Section rétractée.** Les mesures ci-dessous sont exactes sur formes analytiques,
> mais l'estimateur **échoue sur les 5416 courbes expérimentales** — il y est le pire des
> quatre, sur chacune des trois références. Voir §6.6 pour le diagnostic. La section est
> conservée parce que le mécanisme du palier qu'elle décrit reste valide et parce que
> l'écart entre ses résultats synthétiques et expérimentaux est instructif.

Le palier n'est pas un défaut des données mais une propriété physique (§2.1).
L'information sur `Dmax` n'est donc pas dans le minimum du résidu mais dans **l'abscisse
où il chute au niveau du bruit**. Cette quantité ne dépend pas du niveau absolu du
résidu — c'est-à-dire pas de la précision de l'ancrage de Guinier, qui est exactement ce
qui ruine la méthode publiée.

Conditionnement : dispersion de `Dmax` pour une perturbation de ±2 % sur le `Rg`
d'ancrage.

| Cas | Shannon publié | coude |
|---|---:|---:|
| sphère 8 nm | 8,3 % | **1,2 %** |
| sphère 14 nm | 16,9 % | **0,9 %** |
| sphéroïde ε = 0,5 | 28,2 % | **2,7 %** |
| core-shell | 12,2 % | **2,6 %** |
| parabole P(r) | 4,2 % | **0,5 %** |
| sphéroïde ε = 2 | 8,3 % | 8,5 % |

L'estimateur du coude porte un biais systématique de −15 %, corrigé par une constante
multiplicative de **1,1495** ajustée sur un premier jeu de 48 cas, puis **gelée** et
évaluée sur les 52 cas indépendants du §1. Un biais systématique se calibre ; le chaos
de l'argmin ne se calibre pas.

### Réserves, à traiter avant toute intégration

* **La queue de distribution reste mauvaise** : p90 à 26 %, maximum à 55 %. Les pires cas
  sont les objets très allongés (sphéroïde prolate ε = 3), où le coude se brouille et
  l'estimation sous-estime.
* La constante de calibration est empirique et ajustée sur des formes analytiques. Elle
  demande une validation sur données expérimentales réelles avec `Dmax` de référence.
* Le seuil de détection du palier (`factor` = 1,1) est un paramètre libre. Une définition
  sans seuil — intersection des deux asymptotes, ou maximum de courbure — serait plus
  propre et reste à essayer.
* Aucune mesure ne porte sur des données de laboratoire à faible rapport signal/bruit,
  cas que l'article met précisément en avant.

---

## 5. Ce que cela donnerait dans BIFT

La motivation initiale était d'économiser le balayage en `(Dmax, α)`. Mesures sur une
sphère de 10 nm, `npt` = 100, `scan_size` = 27 :

| Scénario | temps | évaluations d'évidence | Dmax obtenu |
|---|---:|---:|---:|
| `auto_bift` tel quel | 4,58 s | 173 | 10,18 |
| `Dmax` figé, α optimisé seul | **0,77 s** | **36** | 10,18 |
| `Dmax` amorcé, toujours raffiné | médiane 1,1× | — | — |

Figer `Dmax` donne **5,9×**, mais impose de vivre avec l'erreur de l'estimateur (5 % en
médiane, 26 % au p90) là où BIFT atteint 1,8 % tout seul. Simplement *amorcer* le
balayage sans le figer ne rapporte que **1,1×** en médiane : Powell refait l'essentiel du
travail quel que soit le point de départ.

Détail de l'amorçage (`Dmax` du coude en graine, raffinement conservé) :

| Cas | Dmax vrai | BIFT seul | t (s) | BIFT amorcé | t (s) | gain |
|---|---:|---:|---:|---:|---:|---:|
| sphère 8 | 8,0 | 8,32 | 6,12 | 8,32 | 5,68 | 1,1× |
| sphère 14 | 14,0 | 13,81 | 8,34 | 14,19 | 2,38 | 3,5× |
| sphéroïde ε=2 | 10,0 | 9,11 | 4,10 | 9,11 | 3,69 | 1,1× |
| core-shell | 10,0 | **12,74** | 4,73 | **10,03** | 6,79 | 0,7× |
| parabole | 12,0 | 12,03 | 2,05 | 12,03 | 1,55 | 1,3× |

> **Bénéfice inattendu, sans rapport avec la vitesse.** Sur la courbe core-shell,
> `auto_bift` seul converge vers 12,74 nm (+27 %, optimum local parasite) alors
> qu'amorcé par le coude il donne 10,03 nm (+0,3 %). L'amorçage vaut peut-être davantage
> comme garde-fou contre les optima locaux que comme accélérateur — mais cela repose sur
> un seul cas et demande à être quantifié.

---

## 6. Validation sur 5416 courbes expérimentales BM29

Les sections 1 à 5 reposent sur des formes analytiques. Cette section les confronte à
`sandbox/2025/` : 5416 courbes expérimentales du même instrument, `q` fixe de 0,085 à
4,96 nm⁻¹ sur 1000 points, mesurées **après** le jeu d'entraînement du réseau — donc un
test véritablement aveugle pour lui, ce que le banc synthétique ne pouvait pas offrir.

Deux références indépendantes :

* **BIFT**, les `*.out` de `free_bift` fournis avec les données ;
* **GNOM**, produit pour cette évaluation via ATSAS 4.1.4 (`autorg` → `datgnom -r Rg`),
  dans `sandbox/2025/atsas/*.out`. 4115 des 5469 courbes aboutissent (75,2 %) ; les
  1354 échecs sont des rejets d'`autorg` (« No Rg found », « Data quality too low »).

Réserve de méthode : les 5416 courbes ne sont pas 5416 mesures indépendantes — on compte
2478 préfixes d'échantillon distincts, certains représentés jusqu'à 210 fois.

### 6.1 Le résultat central : `Dmax` n'est pas une quantité bien déterminée

Avant de juger un estimateur, il faut mesurer l'accord des références entre elles. Sur
les 4098 courbes disposant des deux :

| Comparaison | méd. \|écart\| | p90 | accord à 10 % |
|---|---:|---:|---:|
| BIFT vs GNOM sur `Dmax`, toutes courbes | **27,40 %** | 95,8 % | 20,7 % |
| BIFT vs GNOM sur `Dmax`, courbes que les deux outils acceptent (n = 1417) | **22,10 %** | 60,2 % | — |

Et le contraste avec `Rg`, sur exactement les mêmes 1417 courbes :

| Grandeur et paires d'outils | méd. \|écart\| |
|---|---:|
| `Rg` : `auto_guinier` de freesas vs `autorg` d'ATSAS | **1,27 %** |
| `Rg` : moments du `P(r)` de BIFT vs de GNOM | **4,96 %** |
| `Dmax` : BIFT vs GNOM | **22,10 %** |

**Deux méthodes établies s'accordent sur `Rg` à 5 % et se contredisent sur `Dmax` à
22 %.** Les implémentations ne sont pas en cause — freesas et ATSAS donnent le même `Rg`
à 1,3 %. C'est `Dmax` qui est intrinsèquement mal contraint par des données SAXS : c'est
le point où `P(r)` s'annule, donc une propriété du support choisi et de la
régularisation, là où `Rg` est un moment intégral robuste.

Le désaccord est de plus **systématique** : BIFT rend toujours un `Dmax` plus grand que
GNOM, de +8,5 % à +28,6 % selon la taille.

| `Dmax` GNOM | n | biais BIFT/GNOM | méd. \|écart\| |
|---|---:|---:|---:|
| 0 – 5 nm | 132 | +21,1 % | 22,8 % |
| 5 – 10 nm | 511 | +28,6 % | 28,7 % |
| 10 – 20 nm | 476 | +19,1 % | 20,5 % |
| 20 – 37 nm | 242 | +8,5 % | 12,2 % |

**Conséquence pour toute cette évaluation : viser 5 % sur `Dmax` n'a pas de sens quand
les deux méthodes de référence ne se départagent pas à mieux que 22 %.** Le classement
des estimateurs rapides dépend d'ailleurs de la référence choisie, comme le montre le
paragraphe suivant.

### 6.2 Le classement dépend de la référence — et cela démasque le réseau

Domaine de validité (défini au §6.3), courbes acceptées par les deux outils, n = 1417 :

| Estimateur | vs GNOM | vs BIFT |
|---|---:|---:|
| `3·Rg` (témoin) | **10,43 %** | 20,55 % |
| Shannon publié | 11,28 % | 16,49 % |
| coude (ma variante) | 15,99 % | 28,80 % |
| réseau de neurones | 32,46 % | **14,21 %** |

Le réseau est le meilleur contre BIFT et le pire contre GNOM. L'arithmétique explique
pourquoi :

```
BIFT / GNOM  = 1,1849      (biais systématique mesuré, §6.1)
DNN  / BIFT  = 1,1053
produit      = 1,3096   vs   DNN / GNOM mesuré = 1,3193
```

**Le réseau n'estime pas `Dmax` : il estime le `Dmax` de BIFT**, dont il a appris la
convention et le biais, plus 10 % supplémentaires. Il faut donc le lire comme un
substitut rapide de BIFT, pas comme une mesure indépendante — et le comparer à BIFT est
partiellement circulaire, tandis que le comparer à GNOM le pénalise pour avoir
correctement reproduit ce sur quoi il a été entraîné.

Le juge le moins biaisé est le sous-ensemble où les deux références concordent à 10 %
près (n = 328, soit 23 % du domaine valide) :

| Estimateur | méd. \|écart\| | p90 | biais | <10 % | <20 % | bat le témoin |
|---|---:|---:|---:|---:|---:|---:|
| `3·Rg` (témoin) | **11,03 %** | 32,0 % | −8,8 % | 47,6 % | 78,0 % | — |
| Shannon publié | 11,38 % | 30,4 % | −8,0 % | 46,6 % | 80,2 % | 50,6 % |
| réseau de neurones | 15,13 % | 54,5 % | +14,9 % | 36,0 % | 60,4 % | 42,4 % |
| coude (ma variante) | 18,84 % | 48,5 % | −18,0 % | 19,8 % | 53,0 % | 27,4 % |

**Sur le juge le plus propre, Shannon publié et `3·Rg` sont à égalité** (11,4 % contre
11,0 %) et Shannon ne bat le témoin que sur 50,6 % des courbes — un tirage à pile ou
face. C'est exactement la conclusion du banc synthétique (§1), retrouvée sur données
réelles par un chemin indépendant.

### 6.3 Une limite dure de la méthode : `Dmax ≤ π/q_min`

Le premier canal de Shannon se situe à `q₁ = π/Dmax`. Si `q₁ < q_min`, **il n'est pas
mesuré** — et c'est le terme dominant de la série alternée. Sur ce montage
`q_min` = 0,0852 nm⁻¹, donc la limite est **π/q_min = 36,9 nm**. L'effet est net :

| canaux non mesurés | n | Shannon | coude | réseau | `3·Rg` |
|---|---:|---:|---:|---:|---:|
| 0 (`Dmax` < 36,9 nm) | 3251 | 20,2 % | 33,4 % | 20,6 % | 23,6 % |
| 1 | 601 | 32,7 % | 62,1 % | **10,6 %** | 33,6 % |
| 2 | 11 | 83,6 % | 89,6 % | 22,7 % | 83,1 % |

C'est une propriété de la **mesure**, pas de l'implémentation : aucun réglage ne la
contourne. 84 % des courbes de ce jeu sont dans le domaine valide. Le réseau, lui,
n'est pas concerné — il s'améliore même à 1 canal manquant.

Corollaire pratique : **la méthode de Shannon exige `q_min ≤ π/Dmax`**, à vérifier avant
tout usage, et à documenter si elle est un jour implémentée.

### 6.4 L'ancrage de Guinier reste le facteur limitant

L'ablation du §1 se reproduit sur données réelles. Sur un sous-échantillon de 773
courbes, en remplaçant le `Rg` d'`auto_guinier` par celui de BIFT :

| Variante | méd. \|écart\| vs BIFT |
|---|---:|
| Shannon ancré sur le `Rg` de Guinier | 22,26 % |
| Shannon ancré sur le `Rg` de BIFT | **11,66 %** |
| coude ancré sur le `Rg` de Guinier | 36,49 % |
| coude ancré sur le `Rg` de BIFT | 40,11 % |

L'erreur de Shannon est **divisée par deux** quand l'ancrage est bon : c'est bien lui, et
non le formalisme, qui limite la méthode. Le `q_min` fixe de ce montage donne
`q_min·Rg` > 0,5 pour un tiers des courbes et > 1 pour 407 d'entre elles, où il n'existe
aucune région de Guinier exploitable. Là, tout ce qui est ancré s'effondre et le réseau,
qui n'a besoin d'aucun ancrage, domine :

| `q_min·Rg` > 1 (n = 407) | méd. \|écart\| vs BIFT |
|---|---:|
| réseau de neurones | **12,54 %** |
| `3·Rg` | 43,47 % |
| Shannon publié | 47,34 % |
| coude | 71,10 % |

Que le réseau se passe de fit de Guinier est son avantage structurel le plus solide, et
il se voit précisément là où les données sont les moins favorables.

### 6.5 La fenêtre de balayage exclut la moitié des données réelles

Sur les courbes exploitables, `Dmax/Rg` s'étale de 2,11 à 4,53 (p5–p95), et **seules
51,8 % tombent dans la fenêtre `[2,5 ; 3,5]·Rg`** que prescrit l'article. Hors fenêtre,
la méthode se dégrade comme prévu au §2.3 : 25,08 % pour `Dmax/Rg` > 3,5 contre 15,66 %
à l'intérieur. La critique formulée sur données synthétiques se vérifie donc en
production, à grande échelle.

### 6.6 Rétractation : l'estimateur du coude ne transfère pas

**Ma proposition du §4 échoue sur données réelles et je la retire.** Elle est le pire des
quatre estimateurs sur chacune des trois références (15,99 % vs GNOM, 28,80 % vs BIFT,
18,84 % sur le consensus) et ne bat le témoin que sur 27,4 % des courbes.

La raison est diagnostiquée : son biais n'est pas une constante mais **croît avec la
taille de l'objet**.

| `Dmax` BIFT | biais du coude |
|---|---:|
| 0 – 5 nm | −17,2 % |
| 5 – 10 nm | −28,8 % |
| 10 – 20 nm | −28,7 % |
| 20 – 40 nm | −38,1 % |
| > 40 nm | −64,4 % |

Aucune constante multiplicative ne peut corriger un biais dépendant de la taille ; même
recentré sur sa médiane expérimentale, il reste à 25,7 % de dispersion. La calibration
×1,1495 ajustée au §4 sur des formes analytiques était une illusion de laboratoire : mon
modèle de bruit synthétique était trop propre, et le plancher de résidu réel — plus haut
et structuré — déclenche la détection du palier trop tôt. Le mécanisme du palier (§2.1)
reste exact ; c'est ma façon d'en détecter l'entrée qui ne résiste pas aux données.

### 6.7 Ce que les données réelles corrigent aussi au §3.2

La « limite de domaine du réseau au-delà de 40 nm », mesurée sur des sphères
synthétiques, **n'existe pas sur données réelles** : dans le bin `Dmax` > 40 nm le réseau
est le meilleur des quatre (10,04 % d'écart médian, 49,7 % des courbes à mieux que
10 %). Mes sphères monodisperses à bords francs de 60 à 100 nm sont hors de sa
distribution d'entraînement *et* physiquement non représentatives du BioSAXS ; les gros
objets réels sont flexibles et polydisperses, avec des courbes lisses que le réseau
connaît bien.

En revanche le défaut aux **petits** objets est confirmé et aggravé : sur les 476 courbes
à `Dmax` < 5 nm, le réseau rend 7,48 nm en médiane contre 3,46 nm pour BIFT. Il ne
descend pas sous sa plage d'entraînement — un plancher, pas du bruit. La référence n'est
pas en cause : dans ce bin `χ²` vaut 0,98 et `σ(Dmax)/Dmax` 3,1 %, comparables aux autres
bins.

Leçon de méthode : mon banc synthétique donne la vérité terrain mais avec des formes non
représentatives ; le jeu expérimental donne des formes représentatives mais sans vérité
terrain. **Il fallait les deux**, et l'écart entre leurs verdicts est lui-même une
information.

---

## 7. Recommandation

1. **Ne pas intégrer la méthode telle que publiée.** Sur le juge expérimental le moins
   biaisé elle est à égalité avec `3·Rg`, déjà en place, et le bat sur 50,6 % des
   courbes — un tirage à pile ou face — pour 18 ms par courbe.
2. **Traiter la question des unités en amont, dans `sasio.py`** : lire la déclaration
   d'unité de l'en-tête SASBDB quand elle est présente, appliquer sinon le seuil sur
   `q_max`, et refuser de deviner dans la bande 0,8–1,2 en exigeant `-u/--unit`. C'est
   le correctif au meilleur rapport valeur/effort de toute cette évaluation : il
   bénéficie à toute la chaîne, pas seulement au `Dmax`.
3. **Corriger `dnn.py`** : plancher aux petits objets (il ne descend pas sous ~7 nm,
   §6.7), validation du signe et de l'ordre de grandeur de la sortie, et contrôle de la
   plage de `q`. Documenter aussi ce que le réseau prédit réellement — le `Dmax` de
   BIFT, biais compris (§6.2) — pour que personne ne le prenne pour une mesure
   indépendante.
4. **Abandonner ma variante du coude.** Elle ne transfère pas aux données réelles
   (§6.6). Si l'on veut poursuivre l'idée du palier, il faut une détection d'entrée qui
   ne dépende pas d'un seuil relatif à un plancher de résidu — le plancher réel est trop
   haut et trop structuré.
5. **Ne pas attendre de gain de vitesse décisif sur BIFT.** 1,1× en amorçage simple,
   5,9× seulement au prix d'un `Dmax` figé. Le levier intéressant est la robustesse aux
   optima locaux, à quantifier.
6. **Reconsidérer la question elle-même.** BIFT et GNOM se contredisent sur `Dmax` à
   22 % tout en s'accordant sur `Rg` à 5 %. Avant d'investir dans un estimateur rapide
   de `Dmax`, il vaudrait la peine de décider *quel* `Dmax` on veut estimer, et de
   documenter que la grandeur elle-même n'est pas déterminée à mieux que ~20 % par ce
   type de données. Un intervalle de confiance honnête servirait probablement mieux les
   utilisateurs qu'une valeur ponctuelle plus rapide.

Le formalisme de Shannon reste un acquis réel de l'article : les équations 3 et 5
reproduisent `I(0)` et `Rg` à 0,01–0,5 % près au bon `Dmax`, et les corrections de
troncature divisent l'erreur par cinq à faible `nmax`. C'est l'usage qui en est fait
pour *trouver* `Dmax` qui est mal posé, pas le formalisme — et la difficulté est
d'ailleurs moins dans la méthode que dans la grandeur visée.

---

## Code de l'évaluation

Tout est dans `shannon/`, hors de `src/freesas` : dépendances numpy/scipy seulement, rien
n'est promu dans le paquet avant que les résultats ne le justifient.

| Fichier | Rôle |
|---|---|
| `nsdmax.py` | Implémentation de la méthode publiée : éq. 3, 5, 6, 7, 10–13 et filtre `q`-adaptatif S31–S32 |
| `synthetic.py` | Courbes à vérité terrain exacte (sphère, sphéroïde, core-shell prolate S19–S21, parabole P(r)), auto-testées contre les `Rg` analytiques |
| `benchmark.py` | Banc principal, 455 courbes, 4 estimateurs → `results.csv` |
| `sensitivity.py` | Balayage des paramètres libres, test de conditionnement |
| `knee.py` | Estimateur du coude et comparaison directe |
| `holdout.py` | Jeu de validation indépendant, polydispersité, deux populations |
| `qrange.py` | Domaines de validité en `qmax` et en `Dmax` |
| `unit_detect.py` | Détection Å⁻¹ / nm⁻¹ : seuil de magnitude contre contre-test par le réseau |
| `experimental.py` | Les 5416 courbes BM29 passées aux 4 estimateurs, référence BIFT → `experimental.csv` |
| `analyze_experimental.py` | Découpages par taille, forme, disponibilité de la région de Guinier |
| `run_atsas.py` | Pilote ATSAS `autorg` → `datgnom`, génère `sandbox/2025/atsas/*.out` → `atsas_gnom.csv` |
| `compare_references.py` | Comparaison à deux références indépendantes, plancher de bruit BIFT ↔ GNOM |
| `fc5088.pdf`, `fc5088sup1.pdf` | L'article et ses informations supplémentaires |

Reproduction :

```bash
cd shannon
P=/home/kieffer/.venv/py313/bin/python
$P synthetic.py            # auto-test du generateur
$P benchmark.py            # banc synthetique principal (~10 min)
$P holdout.py              # validation synthetique independante
$P unit_detect.py          # detection d unite
# validation experimentale
$P experimental.py         # 5416 courbes, reference BIFT (~25 min)
$P analyze_experimental.py
$P run_atsas.py            # genere les P(r) GNOM (~60 min, ATSAS requis)
$P compare_references.py   # les deux references
```

ATSAS 4.1.4 est attendu dans `/home/kieffer/ATSAS-4.1.4-1` (variable `ATSAS` positionnée
par `run_atsas.py`).

Données de test : `silx.org/pub/freesas/testdata` via `freesas.test.utilstest`.
Apoferritine SASDFN8 : `https://www.sasbdb.org/media/intensities_files/SASDFN8.dat`.
