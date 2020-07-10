# Changelog

## [0.5](https://github.com/rapidsai/gQuant/tree/0.5) (2020-07-10)

[Full Changelog](https://github.com/rapidsai/gQuant/compare/0.4.1...0.5)

**Implemented enhancements:**

- \[FEA\] csvStockLoader.py and stockNameLoader.py - Use cudf.read\_csv\(\) insteand of pandas.read\_csv\(\) [\#24](https://github.com/rapidsai/gQuant/issues/24)

**Fixed bugs:**

- \[BUG\] Using a UDF via Series.rolling.apply\(\) results in KeyError in numba [\#88](https://github.com/rapidsai/gQuant/issues/88)
- \[BUG\] download\_data.sh seems to do not be in containers anymore [\#66](https://github.com/rapidsai/gQuant/issues/66)

**Closed issues:**

- \[FEA\] Conda resolves too slow in the latest versions of the container [\#67](https://github.com/rapidsai/gQuant/issues/67)
- \[FEA\] Comprehensive refactoring of indicator\_demo.ipynb notebook [\#46](https://github.com/rapidsai/gQuant/issues/46)
- \[FEA\] Rename viz\_graph\(\) to viz\(\), save\_taskgraph\(\) to save\(\) [\#34](https://github.com/rapidsai/gQuant/issues/34)

**Merged pull requests:**

- \[REVIEW\] Fix mortgage e2e example for rapids 0.14. [\#93](https://github.com/rapidsai/gQuant/pull/93) ([avolkov1](https://github.com/avolkov1))
- \[REVIEW\] Update RAPIDS to version 0.14 [\#92](https://github.com/rapidsai/gQuant/pull/92) ([yidong72](https://github.com/yidong72))
- \[REVIEW\]Multiple gpu xgboost - Dask performance fix [\#91](https://github.com/rapidsai/gQuant/pull/91) ([yidong72](https://github.com/yidong72))
- \[REVIEW\]Mutliple GPU xgboost [\#90](https://github.com/rapidsai/gQuant/pull/90) ([yidong72](https://github.com/yidong72))

## [0.4.1](https://github.com/rapidsai/gQuant/tree/0.4.1) (2020-05-26)

[Full Changelog](https://github.com/rapidsai/gQuant/compare/0.4...0.4.1)

**Merged pull requests:**

- \[REVIEW\] hot fix for 0.4 release [\#86](https://github.com/rapidsai/gQuant/pull/86) ([yidong72](https://github.com/yidong72))
- \[REVIEW\] fix the cuIndicator notebook and RSI perf notebook [\#85](https://github.com/rapidsai/gQuant/pull/85) ([yidong72](https://github.com/yidong72))
- Add cuda102 docker support and update version against development branch [\#84](https://github.com/rapidsai/gQuant/pull/84) ([jbaron](https://github.com/jbaron))

## [0.4](https://github.com/rapidsai/gQuant/tree/0.4) (2020-05-19)

[Full Changelog](https://github.com/rapidsai/gQuant/compare/v0.2...0.4)

**Implemented enhancements:**

- \[REVIEW\]Feature adding fractional differencing computation [\#56](https://github.com/rapidsai/gQuant/pull/56) ([yidong72](https://github.com/yidong72))

**Fixed bugs:**

- \[BUG\] Dask computation fails with 0.8 build script [\#28](https://github.com/rapidsai/gQuant/issues/28)

**Closed issues:**

- \[FEA\] Add cuda 10.1.2 support [\#64](https://github.com/rapidsai/gQuant/issues/64)
- \[FEA\] Use RAPIDS 0.9 container in build.sh [\#54](https://github.com/rapidsai/gQuant/issues/54)
- \[FEA\] Rename notebook to notebooks [\#50](https://github.com/rapidsai/gQuant/issues/50)
- \[FEA\] Add Jupyterlab extension to display GPU usage [\#49](https://github.com/rapidsai/gQuant/issues/49)
- \[FEA\] Merge develop branch to master [\#47](https://github.com/rapidsai/gQuant/issues/47)
- \[FEA\] implement the fractional difference operation [\#42](https://github.com/rapidsai/gQuant/issues/42)

**Merged pull requests:**

- \[REVIEW\] merge develop to master and release it as 0.4 [\#82](https://github.com/rapidsai/gQuant/pull/82) ([yidong72](https://github.com/yidong72))
- \[REVIEW\]update to latest version of RAPIDS 0.13 [\#81](https://github.com/rapidsai/gQuant/pull/81) ([yidong72](https://github.com/yidong72))
- fixed the gamma computation error [\#79](https://github.com/rapidsai/gQuant/pull/79) ([doyend](https://github.com/doyend))
- \[REVIEW\]asian barrier option  tutorial [\#77](https://github.com/rapidsai/gQuant/pull/77) ([yidong72](https://github.com/yidong72))
- \[REVIEW\] upgrade to RAPIDS 0.11 [\#76](https://github.com/rapidsai/gQuant/pull/76) ([yidong72](https://github.com/yidong72))
- \[skip ci\] Merge CI Scripts [\#75](https://github.com/rapidsai/gQuant/pull/75) ([avolkov1](https://github.com/avolkov1))
- \[REVIEW\] Add CI scripts and conda recipe [\#74](https://github.com/rapidsai/gQuant/pull/74) ([raydouglass](https://github.com/raydouglass))
- \[WIP\] CUQ-36: fix typechecking nodes multi input dataframes [\#68](https://github.com/rapidsai/gQuant/pull/68) ([avolkov1](https://github.com/avolkov1))
- \[REVIEW\] Upgrade to RAPIDS 0.10 [\#63](https://github.com/rapidsai/gQuant/pull/63) ([yidong72](https://github.com/yidong72))
- \[REVIEW\] stable master merge [\#62](https://github.com/rapidsai/gQuant/pull/62) ([yidong72](https://github.com/yidong72))
- \[REVIEW\]upgrade to RAPIDS 0.9, FIX the rebase problem [\#61](https://github.com/rapidsai/gQuant/pull/61) ([yidong72](https://github.com/yidong72))
- Revert "\[REVIEW\]upgrade to RAPIDS 0.9" [\#59](https://github.com/rapidsai/gQuant/pull/59) ([yidong72](https://github.com/yidong72))
- Revert "\[REVIEW\]upgrade to RAPIDS 0.9" [\#58](https://github.com/rapidsai/gQuant/pull/58) ([avolkov1](https://github.com/avolkov1))
- \[REVIEW\]upgrade to RAPIDS 0.9 [\#57](https://github.com/rapidsai/gQuant/pull/57) ([yidong72](https://github.com/yidong72))
- \[REVIEW\] change the text for notebook 05 [\#55](https://github.com/rapidsai/gQuant/pull/55) ([yidong72](https://github.com/yidong72))
- Fix \#50b - Rename notebook folder to notebooks [\#52](https://github.com/rapidsai/gQuant/pull/52) ([miguelusque](https://github.com/miguelusque))
- Fix \#50 - Rename notebook folder to notebooks [\#51](https://github.com/rapidsai/gQuant/pull/51) ([miguelusque](https://github.com/miguelusque))

## [v0.2](https://github.com/rapidsai/gQuant/tree/v0.2) (2019-08-16)

[Full Changelog](https://github.com/rapidsai/gQuant/compare/v0.1...v0.2)

**Implemented enhancements:**

- \[FEA\] Refactor 04\_portfolio\_trade.ipynb notebook [\#39](https://github.com/rapidsai/gQuant/issues/39)
- \[FEA\] Refactor notebook 01\_tutorial.ipynb [\#35](https://github.com/rapidsai/gQuant/issues/35)
- \[FEA\] Add error message \(or warning\) if replace node does not exist [\#32](https://github.com/rapidsai/gQuant/issues/32)
- \[FEA\] Add new issue templates [\#26](https://github.com/rapidsai/gQuant/issues/26)
- \[FEA\] cuIndicator notebook plot widget is too complicated [\#17](https://github.com/rapidsai/gQuant/issues/17)

**Fixed bugs:**

- \[BUG\] Remove debug info from barPlotNode.py and cumReturnNode.py [\#40](https://github.com/rapidsai/gQuant/issues/40)
- \[BUG\] 04\_portfolio\_trade.ipynb - Number of filtered stocks differs from text [\#23](https://github.com/rapidsai/gQuant/issues/23)

**Merged pull requests:**

- Fix \#17 - cuIndicator notebook plot widget is too complicated \(WIP\) [\#45](https://github.com/rapidsai/gQuant/pull/45) ([miguelusque](https://github.com/miguelusque))
- Fix \#39 - Refactor 04\_portfolio\_trade.ipynb notebook [\#44](https://github.com/rapidsai/gQuant/pull/44) ([miguelusque](https://github.com/miguelusque))
- Merge develop to master [\#43](https://github.com/rapidsai/gQuant/pull/43) ([yidong72](https://github.com/yidong72))
- Fix \#40 - Remove debug info [\#41](https://github.com/rapidsai/gQuant/pull/41) ([miguelusque](https://github.com/miguelusque))
- Update mortgage example using TaskGraph API. [\#38](https://github.com/rapidsai/gQuant/pull/38) ([avolkov1](https://github.com/avolkov1))
- fixed the issue 32 [\#37](https://github.com/rapidsai/gQuant/pull/37) ([yidong72](https://github.com/yidong72))
- Fix \#35 - Refactor 01\_tutorial.ipynb notebook [\#36](https://github.com/rapidsai/gQuant/pull/36) ([miguelusque](https://github.com/miguelusque))
- Fix \#26b - Add new issue templates [\#30](https://github.com/rapidsai/gQuant/pull/30) ([miguelusque](https://github.com/miguelusque))
- Revert "fix \#26 - Add new issues template" [\#29](https://github.com/rapidsai/gQuant/pull/29) ([yidong72](https://github.com/yidong72))
- Fix \#26 - Add new issues template [\#27](https://github.com/rapidsai/gQuant/pull/27) ([miguelusque](https://github.com/miguelusque))
- added workflow class [\#22](https://github.com/rapidsai/gQuant/pull/22) ([yidong72](https://github.com/yidong72))
- Fix \#19b - Combine OS/Cuda versions user input [\#21](https://github.com/rapidsai/gQuant/pull/21) ([miguelusque](https://github.com/miguelusque))
- Fix \#19 - build.sh - Move pip dependencies to conda dependencies [\#20](https://github.com/rapidsai/gQuant/pull/20) ([miguelusque](https://github.com/miguelusque))
- Fix \#13, \#14, \#16 in cuIndicator.ipynb notebook [\#18](https://github.com/rapidsai/gQuant/pull/18) ([miguelusque](https://github.com/miguelusque))
- update the build.sh [\#15](https://github.com/rapidsai/gQuant/pull/15) ([yidong72](https://github.com/yidong72))
- Feature xgb notebook [\#11](https://github.com/rapidsai/gQuant/pull/11) ([yidong72](https://github.com/yidong72))
- CUQ-5: Mortgage example using gQuant. [\#10](https://github.com/rapidsai/gQuant/pull/10) ([avolkov1](https://github.com/avolkov1))
- CUQ-5: Mortgage example using  gQuant. [\#9](https://github.com/rapidsai/gQuant/pull/9) ([avolkov1](https://github.com/avolkov1))
- Feature indicator node [\#8](https://github.com/rapidsai/gQuant/pull/8) ([yidong72](https://github.com/yidong72))
- Feature mulit assets indicator [\#7](https://github.com/rapidsai/gQuant/pull/7) ([yidong72](https://github.com/yidong72))
- Update build.sh [\#6](https://github.com/rapidsai/gQuant/pull/6) ([phogan-nvidia](https://github.com/phogan-nvidia))
- Feature environment [\#5](https://github.com/rapidsai/gQuant/pull/5) ([yidong72](https://github.com/yidong72))

## [v0.1](https://github.com/rapidsai/gQuant/tree/v0.1) (2019-08-13)

[Full Changelog](https://github.com/rapidsai/gQuant/compare/e4a967fc9e3289fdbfa37e7a7b84887579332b42...v0.1)

**Implemented enhancements:**

- \[FEA\] build.sh - Move pip dependencies to conda dependencies [\#19](https://github.com/rapidsai/gQuant/issues/19)

**Fixed bugs:**

- \[BUG\] Update build.sh to 0.7 until issue \#28 is fixed [\#31](https://github.com/rapidsai/gQuant/issues/31)
- \[BUG\] cuIndicator.ipyng - Wrong series names [\#16](https://github.com/rapidsai/gQuant/issues/16)
- \[BUG\] cuIndicator.ipynb - Runtime error in cell \#3 - Missing file [\#14](https://github.com/rapidsai/gQuant/issues/14)
- \[BUG\] cuIndicator.ipynb - Incorrect path to dataset [\#13](https://github.com/rapidsai/gQuant/issues/13)

**Merged pull requests:**

- Revert "gQuant34 - Update build.sh to make use of RAPIDS v0.8 container" [\#33](https://github.com/rapidsai/gQuant/pull/33) ([yidong72](https://github.com/yidong72))
- gQuant34 - Update build.sh to make use of RAPIDS v0.8 container [\#12](https://github.com/rapidsai/gQuant/pull/12) ([miguelusque](https://github.com/miguelusque))
- Synch master with develop [\#4](https://github.com/rapidsai/gQuant/pull/4) ([avolkov1](https://github.com/avolkov1))
- added unit tests for the cuindicator [\#3](https://github.com/rapidsai/gQuant/pull/3) ([yidong72](https://github.com/yidong72))
- CUQ-21: Improving tutorials for gQuant [\#2](https://github.com/rapidsai/gQuant/pull/2) ([avolkov1](https://github.com/avolkov1))
- Add download script and instructions in the readme [\#1](https://github.com/rapidsai/gQuant/pull/1) ([yidong72](https://github.com/yidong72))



\* *This Changelog was automatically generated by [github_changelog_generator](https://github.com/github-changelog-generator/github-changelog-generator)*
