# Sandbox for agent-written Python: no credentials, no network, read-only data and synlib.
# Build once:  python -m synaptic sandbox build   (or: docker build -t synaptic-sandbox:4 sandbox/)
FROM python:3.13-slim
RUN pip install --no-cache-dir numpy pandas scipy statsmodels matplotlib netCDF4 h5py scikit-learn
RUN useradd -m -u 10001 agent
USER agent
WORKDIR /work
