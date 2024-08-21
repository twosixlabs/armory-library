# Armory open-source strategy

This document is submitted to JATIC leadership to explain our strategy and
rationale for releasing the JATIC development of Armory to open access.

## GARD-Armory

The GARD version of Armory application has been open-source since its inception at
https://github.com/twosixlabs and a `pip` installable at
https://pypi.org/project/armory-testbed/. GARD Armory is currently at version 0.18.2 and
will be supported through the duration of the GARD research program into 2024. Aside
from bug fixes and minor point-releases, v0.18.x will likely be the last releases of
GARD-Armory.

## JATIC-Armory

When we began work in 2023 on JATIC-Armory we kept it on the closed server
https://gitlab.jatic.net/ because the CDAO team had not fully determined how the JATIC
project should handle open-source.

The JATIC-Armory repository began as a direct clone of GARD-Armory with the expectation
that it would diverge radically from its origin. It has.

By the end of 2023, with the acceptance of this strategy by JATIC leadership,
JATIC-Armory will supplant the GARD-Armory on Github. That is, JATIC-Armory will
become the next release of Armory. With this transition, we are adopting "Ubuntu"
numbering for our releases where Armory v23.12 will be the December 2023 release.

The pip installable name no longer makes sense for JATIC-Armory, so versions 23.x and
following will be available under the package name
[armory-library](https://pypi.org/project/armory-library/).

For convenience among JATIC performers, we have already published JATIC-Armory
to the Python Package Index as [charmory](https://pypi.org/project/charmory/).
The name "charmory" is a work-in-progress name for the nascent JATIC-armory
to reduce confusion. With the strategy elaborated here, particularly the
change of package name from `armory-testbed` to `armory-library` the scaffold
name will disappear over the coming months.

### Release cadence

For the benefit of fellow JATIC performers, we've committed to one versioned
release per Increment (quarter) although we may well be making tagged releases
once a month. We will tag bug-fix releases (for example, v23.7.2) as development
and users require.

We have set up Gitlab CI such that the tagging of a release causes it to be
built, tested, and published to PyPI automatically.

By virtue of being open source, interested repository watchers will have
daily access to our development work, as their needs require.

### Target platforms

As a pure Python library with no binary elements, Armory will run on any
platform supported by Python. By virtue of our import of PyTorch, the
operation of Armory does require binary support as well provided by PyTorch
itself.

### Synchronization strategy

When we bring JATIC Armory to Github we intend to synchronize the jatic.gitlab
and github repositories at least nightly, put hopefully on every developer
push to the repository.

There is a mechanical concern that we will need to address: mirroring between
Gitlab and Github is afforded and can operate as a pull from gitlab to github
or a push from gitlab to github; bidirectional mirroring is supported
but is [expected to raise consistency issues][gitlab-bidirectional].

It is our belief that the proper structure is to treat the public-facing
Github server as authoritative with the gitlab.jatic server using pull mirroring
from Github.com. Although this is a polling activity, Gitlab is structured
to check with Github [once per minute][gitlab-pull] so as a practical
matter gitlab.jatic will track quite closely to the authoritative repository.

This pull strategy also has the advantage that JATIC created issues will _not_
propagate to Github, which allows our private planning and monitoring
to be the exclusive provenience of JATIC so that internal notes don't
accidentally bleed out to the public-facing Github. However, issues and
pull requests submitted by the general public to Github will flow through
to Gitlab.

### Reviewing pull requests from the internet

Armory team has 3+ year's experience fielding issues and pull-requests
to our open-source Github repository. We expect our procedures and practices
to continue to serve us well. No one except TwoSix personnel are able
to merge pull requests, and we explicitly prohibit pushes by anyone
to the `master` and `development` branches on the Github Armory repo.
As a result of these practices, not even a TwoSix staffer can maliciously
or accidentally pollute our protected branches by themselves, with external
contributors facing an even higher bar.

As a matter of TwoSix policy, the engineers who approve an external pull-request
are responsible for the entire content of that PR as if they wrote it themselves,
and that professional responsibility has ensured that Armory has been in
our exclusive control since the first commit in 2019.

### Open source license

Armory has been published, since its inception, under [The MIT
License](https://opensource.org/license/mit/) and will continue to be so licensed.

### Mechanics

Armory has always borne a `CONTRIBUTING.md` and will continue. We
have pre-commit hooks available to contributors, but we do not enforce those
upon push. However, our CI system, which activates on every commit push
does run Black, Isort, Flake8, and MyPy along with Bandit security scanning.



[gitlab-bidirectional]: https://gitlab.jatic.net/help/user/project/repository/mirror/bidirectional.md
[gitlab-pull]: https://gitlab.jatic.net/help/user/project/repository/mirror/pull.md
