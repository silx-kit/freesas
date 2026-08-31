# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

FreeSAS: outils Python (+ Cython) d'analyse de diffusion des rayons X aux petits angles (SAXS/BioSAXS), développés à l'ESRF. Référence : [J. Synchrotron Rad. (2022). 29, 1318-1328](https://scripts.iucr.org/cgi-bin/paper?ju5045). Licence MIT.

Le package est un **layout `src/`** (`src/freesas/`) construit avec **meson-python** — il ne peut **pas** être importé depuis les sources : `src/freesas/__init__.py` lève `RuntimeError` si `freesas.version` n'a pas été généré par le build.

## Commandes

```bash
# Build local + exécution d'un script sans installer (patche PYTHONPATH à la volée)
./bootstrap.py free_rg fichier.dat
./bootstrap.py ipython

# Tests unitaires : construit d'abord le projet via meson, puis lance freesas.test.suite
python run_tests.py
python run_tests.py --installed            # teste la version installée au lieu de builder
python run_tests.py -c                     # rapport de couverture (nécessite coverage, lxml)
python run_tests.py -v                     # verbeux (répéter pour plus de détails)

# Un seul test / module (chemin d'import complet, résolu par unittest.loadTestsFromNames)
python run_tests.py freesas.test.test_dnn
python run_tests.py freesas.test.test_autorg.TestAutoRg.test_autorg

# Tests end-to-end : lancent les exécutables installés en sous-process ⇒ nécessitent `pip install .`
pip install .
cd e2etest && python e2etest.py

# Build/installation
pip install .                              # meson-python, compile les extensions Cython
meson setup build && meson install -C build --destdir .

# Documentation Sphinx (build/sphinx)
./build-doc.py

# Lint (CI)
pylint $(git ls-files '*.py')
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

`run_tests.py` et `build-doc.py` appellent tous deux `bootstrap.build_project()`, qui fait `meson setup build` puis `meson install --destdir .` et retourne `build/lib/python3.X/site-packages`.

Les données de test **ne sont pas dans le dépôt** : `src/freesas/test/utilstest.py` les télécharge depuis `http://www.silx.org/pub/freesas/testdata` via `get_datafile(name)` (cache dans `build/`, surchargeable par la variable d'environnement `FREESAS_TESTDATA`). La liste est dans `all_testdata.json`.

## Architecture

### Couches

1. **Noyau Cython** (`src/freesas/ext/*.pyx`) — code chaud : `_autorg.pyx` (fit de Guinier, `autoRg`, exceptions `InsufficientDataError`/`NoGuinierRegionError`), `_bift.pyx` (classe `BIFT`), `_cormap.pyx` (`measure_longest`), `_distance.pyx` (NSD entre modèles). Chaque `.pyx` est déclaré comme `py.extension_module` dans `src/freesas/ext/meson.build` ; **ajouter un `.pyx` implique d'éditer ce meson.build**.
2. **Modules scientifiques** (`src/freesas/*.py`) — enrobent les extensions : `autorg.py` (`auto_gpa`, `auto_guinier`, réexporte `autoRg`), `bift.py` (`auto_bift`), `cormap.py` (`gof`), `invariants.py` (Porod, Vc, Rambo-Tainer), `model.py`/`align.py`/`average.py` (modèles PDB 3D, alignement type supcomb, moyennage), `transformations.py` (bibliothèque de transformations homogènes, code externe volumineux), `plot.py` (figures matplotlib), `dnn.py` (inférence NumPy pure d'un réseau dense Keras).
3. **Applications CLI** (`src/freesas/app/*.py`) — chaque module expose `build_parser()` + `main()`, référencés dans `[project.scripts]` de `pyproject.toml`. Ajouter une application = nouveau module + entrée dans `pyproject.toml` **et** dans `src/freesas/app/meson.build`.

### Conventions structurantes

- **`containers.py`** définit tous les types de résultats sous forme de `namedtuple` (`RG_RESULT`, `FIT_RESULT`, `RT_RESULT`, `EvidenceResult`, `StatsResult`…), avec des `__repr__` patchés après coup qui produisent l'affichage « native » des CLI. Modifier ce `__repr__` change la sortie utilisateur et casse les tests e2e.
- **`sas_argparser.py`** : `SASParser` enrobe `argparse.ArgumentParser` en ajoutant `-v/--verbose` et `-V/--version` ; `GuinierParser` le compose avec les arguments standards des apps Guinier (fichiers, `-o/--output`, `-f/--format` parmi native/csv/ssf, `-u/--unit` nm ou Å). Les CLI ne créent jamais d'`ArgumentParser` directement.
- **`fitting.py`** : `run_guinier_fit(fit_function, parser, logger)` est le pipeline partagé par `free_rg`, `free_gpa` et `free_guinier`. Les trois apps ne diffèrent que par la fonction de fit passée en argument (`autoRg`, `auto_gpa`, `auto_guinier`) et le texte d'aide. Toute nouvelle app Guinier doit suivre ce schéma, cf. `src/freesas/app/auto_gpa.py`.
- **`sasio.py`** : point d'entrée unique de lecture (`load_scattering_data`, `parse_ascii_data`, `convert_inverse_angstrom_to_nanometer`). L'unité interne est le **nm⁻¹** ; la conversion depuis l'Å se fait à l'entrée du pipeline.
- **`resources/`** : les fichiers de données (dont `keras_models/Rg+Dmax.keras`) s'accèdent **uniquement** via `resource_filename()` (compatible zip/frozen/packaging distro), jamais par chemin relatif. Nouveau fichier de ressource ⇒ l'ajouter au `meson.build` correspondant.

### Tests

`src/freesas/test/test_all.py` agrège manuellement les `suite()` de chaque module de test ; `e2etest/e2etest.py` fait de même pour les tests e2e. **Un nouveau fichier de test doit être ajouté explicitement à cet agrégateur** (et à `src/freesas/test/meson.build`), sinon il n'est jamais exécuté. Les tests unitaires exposent tous une fonction `suite()` construite à la main — le style est unittest, pas pytest.

### Versionnage

`version.py` (à la racine, installé dans le package sous `freesas/version.py`) est la source unique de vérité : `MAJOR/MINOR/MICRO/RELEV/SERIAL`. `meson.build` récupère la version via `run_command(['version.py', '--wheel'])`. Ne pas éditer la version ailleurs.

### Compatibilité

`requires-python = '>=3.7'` et la CI Ubuntu teste 3.7 → 3.12 : éviter la syntaxe postérieure à 3.7 (pas de `match`, pas de `X | Y` en annotation à l'exécution). Les f-strings sont utilisées partout.

## Branche courante

`dense_neural_network` : ajout de `dnn.py` (réimplémentation NumPy de l'inférence d'un modèle Keras dense, lu directement depuis l'archive `.keras` — `config.json` + `model.weights.h5` — sans dépendance TensorFlow) pour prédire Rg et Dmax. `preprocess()` normalise I par son max et interpole sur 1024 points q réguliers dans [0, 4] nm⁻¹.
