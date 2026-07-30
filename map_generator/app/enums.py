from enum import Enum


class COMMAND(Enum):
    GET_PROTOCOL_VERSION = "get-protocol-version"
    LIST_DOCUMENTS = "list-documents"
    LIST_SESSIONS = "list-sessions"
    LIST_SESSION_TARGETS = "list-session-targets"
    CREATE_TARGET_AT_LOCATION = "create-target-at-location"
    CREATE_SAMPLE = "create-sample"
    SELECT_TARGET_IN_SESSION = "select-target-in-session"
    SET_STREAM_OPTION = "set-stream-option"


class STREAM(Enum):
    SESSION_CROSSHAIRS_MOVED = "session-crosshairs-moved"
    TARGET_SELECTED = "target-selected"
    SAMPLE_CREATION = "sample-creation"
    SAMPLE_EMG = "sample-emg"
    SESSION_POLARIS_UPDATE = "session-polaris-update"
    SESSION_TTL_TRIGGERS = "session-ttl-triggers"


class COORDINATE_SYSTEM(Enum):
    BRAINSIGHT = "Brainsight"
    WORLD = "World"


class ERROR_TYPE(Enum):
    NoError = 0
    PacketInvalidJSON = 100
    PacketNameInvalid = 101
    PacketUUIDInvalid = 102
    RequiredFieldMissing = 103
    WrongType = 104
    TooLong = 105
    TooShort = 106
    InvalidCombination = 107
    NoDocuments = 201
    MoreThanOneDocument = 202
    NoActiveSession = 301
    NoActiveSessionWithName = 302
    PerformStepNotLoaded = 303
    MatrixSizeNot4x4 = 401
    CrazyFloatingPoint = 402
    NonInvertibleMatrix = 403
    NonRigidMatrix = 404
    CoordinateSystemUnknown = 501
    GeneralSampleCreationFailure = 601
    InvalidStreamName = 801
    NoTargetWithName = 901
    NoTargetWithIndexPath = 902
