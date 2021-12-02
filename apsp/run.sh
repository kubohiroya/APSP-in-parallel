java -Xmx16g -cp target/classes:$HOME/.m2/repository/net/java/dev/jna/jna/5.8.0/jna-5.8.0.jar:$HOME/.m2/repository/net/java/dev/jna/jna-platform/5.8.0/jna-platform-5.8.0.jar -Djna.library.path=$HOME/APSP-in-parallel/apsp/target/classes org.idegsm.AllPairsShortestPathsTestResolver $1 $2 $3


