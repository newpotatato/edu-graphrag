FROM busybox:1.37.0-uclibc AS probe

FROM grafana/alloy:v1.10.2 AS alloy
COPY --from=probe /bin/busybox /bin/busybox

FROM otel/opentelemetry-collector:0.130.1 AS collector
COPY --from=probe /bin/busybox /bin/busybox

FROM grafana/tempo:2.8.2 AS tempo
COPY --from=probe /bin/busybox /bin/busybox
USER root
RUN ["/bin/busybox", "mkdir", "-p", "/var/tempo"]
RUN ["/bin/busybox", "chown", "10001:10001", "/var/tempo"]
USER 10001:10001
