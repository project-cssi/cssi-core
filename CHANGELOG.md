<a name="cssi-0.1a1"></a>
## [cssi-0.1a1](https://github.com/brionmario/cssi-core/compare/bde8b017c85c4da272746827ce4ae13f5d4e6430...cssi-0.1a1) (2019-05-03)

### Bug Fixes
- **config:** fix weight percentages loading issue ([30ab89c](https://github.com/brionmario/cssi-core/commit/30ab89c1bad8e37e6dbecc0175813aec2664ac0a))
- **core:** fix celery hanging on model.predic issue :bug: ([d243950](https://github.com/brionmario/cssi-core/commit/d2439507713ead3519db304581e938eca2e1d802))
- **questionnaire:** fix string calculation bug :bug: ([23e1cfd](https://github.com/brionmario/cssi-core/commit/23e1cfdfa91d4c08ee2eb8b965f184e03ef94039))

### Code Refactoring
- **core:** implement proper inheritance and refactor contributors ([5f96909](https://github.com/brionmario/cssi-core/commit/5f969093f89afe31b0cd8190880dc75055adfeb8))
- **core:** move CSSIContributor interface to core module :truck: ([d3e651b](https://github.com/brionmario/cssi-core/commit/d3e651b3e4b23a54f17c552699ab4532ba914d7d))
- **core:** remove unused abstract method from contrib base ([79f1251](https://github.com/brionmario/cssi-core/commit/79f1251d8eb7f5221ffc1418c448fffb8a6988f7))
- **core:** remove unused misc.py :truck: ([332b550](https://github.com/brionmario/cssi-core/commit/332b5502f2bcfefc22c19f03e6d99026c8a1204e))
- **core:** rename and refactor contributors :recycle: ([076ca0a](https://github.com/brionmario/cssi-core/commit/076ca0ac2f4f4f18777f1b5f110b83b9700bcc2b))
- **core:** rename contributor abstract methods ([33f38b0](https://github.com/brionmario/cssi-core/commit/33f38b041b23ac50818ed140bbf23ebd84770cdf))
- **core:** update all the double quotes to single ([2f107b5](https://github.com/brionmario/cssi-core/commit/2f107b5a76bd5c37864bd635768a9c4e98fc16e6))
- **latency:** reactor head pose code ([50302d5](https://github.com/brionmario/cssi-core/commit/50302d5acef808d3f851f1eb6d6c02a0ea580373))
- **questionnaires:** refactoring to cater name change ([2fc6959](https://github.com/brionmario/cssi-core/commit/2fc695923e25540e17ec1af87c56e6345b5460ab))
- **sentiment:** refactor paths :recycle: ([a2853d3](https://github.com/brionmario/cssi-core/commit/a2853d3ce695f00a2c183999380e539a6f7126d6))
- **sentiment:** remove mistakenly versioned lines ([cd957c1](https://github.com/brionmario/cssi-core/commit/cd957c13bee56fcfa24a1e1202f4bc446fbc48b6))
- **utils:** change euler function's param name ([efb3b98](https://github.com/brionmario/cssi-core/commit/efb3b9846ee35630cdd5fc907e5345b67382151a))
- **utils:** rename calculate angle diff function ([b1696ca](https://github.com/brionmario/cssi-core/commit/b1696cac243de4e268f4d3034da87dbc820bffaf))
- **utils:** rename image_processing util file :truck: ([ee98335](https://github.com/brionmario/cssi-core/commit/ee983350d3fa73d148226ba1318fc318d1405c87))
- move all the models and classifiers to data folder :truck: ([e132f84](https://github.com/brionmario/cssi-core/commit/e132f8408c07323df36a31198985935849de1770))
- refactor latency and sentiment modules to use util functions ([02ce340](https://github.com/brionmario/cssi-core/commit/02ce3403578da22264c037df0a99d777f4117d36))
- remove unused exceptions ([ce51004](https://github.com/brionmario/cssi-core/commit/ce51004ee8afc2f748e2b1be49aa297fe80223ce))

### Features
- **config:** add config loading support :sparkles: ([ae28c9b](https://github.com/brionmario/cssi-core/commit/ae28c9be2bbbb8585ecef5e6241d42e7191c1e98))
- **core:** add logging support :sparkles: ([5893aae](https://github.com/brionmario/cssi-core/commit/5893aae84fb6014004e9bf69856dbe5fa044576a))
- **core:** implement initial version of the CSSI algorithm ([a11bdda](https://github.com/brionmario/cssi-core/commit/a11bdda77600f68e526e0a8e6c7573acc58f58aa))
- **latency:** add latency scoring functionality :sparkles: ([db205c2](https://github.com/brionmario/cssi-core/commit/db205c2e8c0edbb15f54d524c36939285272b619))
- **latency:** add pst calculation logic :sparkles: ([6694a49](https://github.com/brionmario/cssi-core/commit/6694a49fcf72db4971c146a266f9e49d8fbf1928))
- **latency:** move from cascade face detection to dnn :sparkles: ([ebf1109](https://github.com/brionmario/cssi-core/commit/ebf110911220acdfddc75d467247e002bb259197))
- **questionnaire:** implement questionnaire score calculation module ([82c7f78](https://github.com/brionmario/cssi-core/commit/82c7f7806ec2042b5ce24cb26da5505d7f8702d3))
- **sentiment:** add final score generation logic :sparkles: ([c0000ad](https://github.com/brionmario/cssi-core/commit/c0000ad77dcef8ad5a3a561a20f1218d18865b70))
- **sentiment:** implement sentiment recognition feature :sparkles: ([8e341ad](https://github.com/brionmario/cssi-core/commit/8e341ad2d50666b29c0fb01e58ee639403715479))
- **sentiment:** move from cascade face detection to dnn :sparkles: ([d9131d7](https://github.com/brionmario/cssi-core/commit/d9131d72bd50c0937884fb2d9d0fb0fe1a13acc1))


