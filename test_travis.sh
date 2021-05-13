echo $ARCH
if [ "$ARCH" = "AARCH64" ]
then
  docker run --rm --privileged multiarch/qemu-user-static:register --reset
  docker build -t ion-arm64 . -f Dockerfile.aarch64
else
  echo $ARCH
  dub test --arch=$ARCH --build=unittest-dip1000
  dub test --arch=$ARCH --build=unittest-cov
fi