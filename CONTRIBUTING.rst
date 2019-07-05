Contributing to CSSI-Core
=========================

We would love for you to contribute to CSSI core library and help make
it even better than it is today!. As a contributor, here are the
guidelines for you to follow:

-  `Commit Message Guidelines`_
-  `Branch Naming Conventions`_

Commit Message Guidelines
-------------------------

We have used ``semantic git commits`` through out the application and
would like to keep them consistent. Please follow the following
specified rules when committing your code.

Click `here`_ to learn more about Semantic Git Commits.

Commit Message Format
~~~~~~~~~~~~~~~~~~~~~

Each commit message consists of a **header**, a **body** and a
**footer**. The header has a special format that includes a **type**, a
**scope** and a **subject**:

::

   <type>(<scope>): <subject>
   <BLANK LINE>
   <body>
   <BLANK LINE>
   <footer>

The **header** is mandatory and the **scope** of the header is optional.

Any line of the commit message cannot be longer 100 characters! This
allows the message to be easier to read on GitHub as well as in various
git tools.

The footer should contain a reference to a bug(issue) if any.

Samples:

::

   docs(readme): update readme

::

   build(bazel): modify bazel build script

   With this change, bazel will optimize the build artifacts.

   Fixes #125
   Closes #168

   PR Close #456


Emojis
~~~~~~

Feel free to spice up the git messages with ``emojis``. Use the `gitmoji`_ guide by Carlos Cuesta to create awesome commits.

Samples:

::

   docs: update README :memo:

   refactor(docs): remove RELEASE guidelines :fire:

   fix(build): fix bazel build issue :bug:

Revert
~~~~~~

If the commit reverts a previous commit, it should begin with
``revert:``, followed by the header of the reverted commit. In the body
it should say: ``This reverts commit <hash>.``, where the hash is the
SHA of the commit being reverted.

Type
~~~~

Must be one of the following:

-  **build**: Changes that affect the build system or external
   dependencies (example scopes: gulp, broccoli, npm)
-  **ci**: Changes to our CI configuration files and scripts (example
   scopes: Circle, BrowserStack, SauceLabs)
-  **docs**: Documentation only changes
-  **feat**: A new feature
-  **fix**: A bug fix
-  **perf**: A code change that improves performance
-  **refactor**: A code change that neither fixes a bug nor adds a
   feature
-  **style**: Changes that do not affect the meaning of the code
   (white-space, formatting, missing semi-colons, etc)
-  **test**: Adding missing tests or correcting existing tests

Scope
~~~~~

The scope should be the name of the npm package affected (as perceived
by the person reading the changelog generated from commit messages.

The following is the list of supported scopes:

-  **core**

-  **latency**

-  **sentiment**

-  **questionnaire**

-  **plugins**

-  **config**

-  **common**

-  **vcs**

-  **setup**

-  none/empty string: useful for ``style``, ``test`` and ``refactor``
   changes that are done across all packages
   (e.g. ``style: add missing semicolons``)

Subject
~~~~~~~

The subject contains a clear description of the change:

-  use the imperative, present tense: “change” not “changed” nor
   “changes”
-  don’t capitalize the first letter
-  no dot (.) at the end

Body
~~~~

Just as in the **subject**, use the imperative, present tense: “change”
not “changed” nor “changes”. The body should include the motivation for
the change and contrast this with previous behavior.

Footer
~~~~~~

The footer should contain any information about **Breaking Changes** and
is also the place to reference GitHub issues that this commit
**Closes**.

**Breaking Changes** should start with the word ``BREAKING CHANGE:``
with a space or two newlines. The rest of the commit message is then
used for this.

Branch Naming Convention
------------------------

Please follow the following convention when creating new branches.

::

   <type>/<name>

Types
~~~~~

.. raw:: html

   <table>
      <thead>
         <tr>
            <th>Prefix</th>
            <th>Use case</th>
         </tr>
      </thead>
      <tbody>
         <tr>
            <td>feature</td>
            <td>New feature</td>
         </tr>
         <tr>
            <td>fix</td>
            <td>Code change linked to a bug</td>
         </tr>
         <tr>
            <td>hotfix</td>
            <td>Quick fixes to the codebase</td>
         </tr>
         <tr>
            <td>release</td>
            <td>Code-base releases</td>
         </tr>
      </tbody>
   </table>

Name
~~~~

Always use dashes to separate words, and keep it short.

Examples
''''''''

::

   feature/config-support
   hotfix/upload-size
   fix/incorrect-upload-progress
   release/1.0.x

.. _Commit Message Guidelines: #commit
.. _Branch Naming Conventions: #branch-naming
.. _here: http://karma-runner.github.io/0.10/dev/git-commit-msg.html
