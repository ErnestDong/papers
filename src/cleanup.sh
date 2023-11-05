find img/ build/ lib/ -type f | xargs rm
find . -type f -maxdepth 1 | grep -v "git" | xargs rm
find src -type f -maxdepth 1 | grep -v "cleanup.sh" | xargs rm
